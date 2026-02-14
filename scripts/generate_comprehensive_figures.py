#!/usr/bin/env python3
"""
PRISM4D Comprehensive Figure Generator

Generates publication-quality visualizations from nhs-analyze-pro output:
  - Figure 11: Burst Event Timeline
  - Figure 12: Confidence Enhancement Visualization
  - Figure 13: Chemical Environment Heatmap
  - Additional: Wavelength Distribution, Selectivity Analysis

Usage:
    python generate_comprehensive_figures.py /path/to/output_dir

Expects these files in output_dir:
    - comprehensive_report.json
    - cryptic_sites.json
"""

import json
import sys
import os
import warnings
from pathlib import Path
from collections import defaultdict

# Suppress harmless 3D import warning (we don't use 3D plots)
warnings.filterwarnings('ignore', message='.*Unable to import Axes3D.*')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Color schemes
WAVELENGTH_COLORS = {
    '250': '#e74c3c',   # S-S - Red
    '258': '#e67e22',   # PHE - Orange
    '265': '#f1c40f',   # General - Yellow
    '274': '#2ecc71',   # TYR - Green
    '280': '#3498db',   # TRP - Blue
    '290': '#9b59b6',   # Thermal - Purple
}

CHROMOPHORE_LABELS = {
    '250': 'S-S (250nm)',
    '258': 'PHE (258nm)',
    '265': 'General (265nm)',
    '274': 'TYR (274nm)',
    '280': 'TRP (280nm)',
    '290': 'Thermal (290nm)',
}

CONFIDENCE_COLORS = {
    'HIGH': '#27ae60',
    'MEDIUM': '#f39c12',
    'LOW': '#95a5a6',
}


def load_data(output_dir):
    """Load all JSON data from output directory."""
    output_dir = Path(output_dir)

    with open(output_dir / 'comprehensive_report.json') as f:
        report = json.load(f)

    with open(output_dir / 'cryptic_sites.json') as f:
        sites = json.load(f)

    return report, sites


def figure_11_burst_timeline(sites, report, output_dir):
    """
    Figure 11: Burst Event Timeline

    Frame-by-frame spike intensity plot highlighting burst events (>150 spikes)
    and color-coded by dominant wavelength.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1])

    # Collect frame data across all HIGH/MEDIUM sites
    frame_spikes = defaultdict(lambda: {'total': 0, 'wavelengths': defaultdict(int)})
    burst_frames = []

    for site in sites:
        if site['category'] not in ['HIGH', 'MEDIUM']:
            continue
        if 'tier2' not in site or not site['tier2']:
            continue

        for ft in site['tier2']['frame_timeseries']:
            frame = ft['frame']
            spikes = ft['spikes']
            frame_spikes[frame]['total'] += spikes

            for wl, count in ft.get('wavelengths', {}).items():
                frame_spikes[frame]['wavelengths'][wl] += count

            # Track burst events
            if spikes >= 150:
                burst_frames.append({
                    'frame': frame,
                    'spikes': spikes,
                    'site_id': site['id'],
                    'wavelength': site.get('dominant_wavelength'),
                })

    if not frame_spikes:
        print("  Warning: No frame data found for burst timeline")
        plt.close()
        return

    # Sort frames
    frames = sorted(frame_spikes.keys())
    totals = [frame_spikes[f]['total'] for f in frames]

    # Top plot: Spike intensity timeline
    ax1 = axes[0]
    ax1.fill_between(frames, totals, alpha=0.3, color='steelblue')
    ax1.plot(frames, totals, color='steelblue', linewidth=1.5, label='Total spikes')

    # Mark burst events
    burst_threshold = 150
    ax1.axhline(y=burst_threshold, color='red', linestyle='--', alpha=0.5,
                label=f'Burst threshold ({burst_threshold})')

    # Highlight burst frames with markers
    burst_x = [b['frame'] for b in burst_frames]
    burst_y = [b['spikes'] for b in burst_frames]
    burst_colors = [WAVELENGTH_COLORS.get(str(int(b['wavelength'])) if b['wavelength'] else '290', '#95a5a6')
                    for b in burst_frames]

    ax1.scatter(burst_x, burst_y, c=burst_colors, s=100, zorder=5,
                edgecolors='black', linewidth=1, label='Burst events')

    ax1.set_xlabel('Frame', fontsize=12)
    ax1.set_ylabel('Spike Count', fontsize=12)
    ax1.set_title('Burst Event Timeline - Spike Intensity Over Trajectory', fontsize=14)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Bottom plot: Wavelength distribution per frame (stacked)
    ax2 = axes[1]
    wavelengths = ['250', '258', '265', '274', '280', '290']

    # Build stacked data
    wl_data = {wl: [] for wl in wavelengths}
    for f in frames:
        for wl in wavelengths:
            wl_data[wl].append(frame_spikes[f]['wavelengths'].get(wl, 0))

    # Stacked area plot
    bottom = np.zeros(len(frames))
    for wl in wavelengths:
        values = np.array(wl_data[wl])
        ax2.fill_between(frames, bottom, bottom + values,
                        color=WAVELENGTH_COLORS[wl], alpha=0.7,
                        label=CHROMOPHORE_LABELS[wl])
        bottom += values

    ax2.set_xlabel('Frame', fontsize=12)
    ax2.set_ylabel('Spikes', fontsize=12)
    ax2.set_title('Wavelength Distribution Over Time', fontsize=11)
    ax2.legend(loc='upper right', ncol=3, fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path(output_dir) / 'Figure11_burst_timeline.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def figure_12_confidence_enhancement(sites, report, output_dir):
    """
    Figure 12: Confidence Enhancement Visualization

    Scatter plot showing entropy vs confidence, with markers for:
    - Sites promoted by chromophore weighting
    - Burst-enhanced sites
    - Quadrant analysis
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left plot: Entropy vs Confidence scatter
    ax1 = axes[0]

    entropies = [s.get('wavelength_entropy', 0.5) for s in sites[:100]]
    confidences = [s['overall_confidence'] for s in sites[:100]]
    categories = [s['category'] for s in sites[:100]]
    max_bursts = [s.get('max_single_frame_spikes', 0) for s in sites[:100]]

    # Color by category, size by burst intensity
    colors = [CONFIDENCE_COLORS.get(cat, '#95a5a6') for cat in categories]
    sizes = [30 + (burst / 10) for burst in max_bursts]

    scatter = ax1.scatter(entropies, confidences, c=colors, s=sizes,
                         alpha=0.6, edgecolors='black', linewidth=0.5)

    # Quadrant lines
    ax1.axhline(y=0.70, color='red', linestyle='--', alpha=0.5)
    ax1.axvline(x=0.70, color='blue', linestyle='--', alpha=0.5)

    # Quadrant labels
    ax1.text(0.35, 0.75, 'SELECTIVE\n+ HIGH CONF', ha='center', va='center',
            fontsize=9, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    ax1.text(0.85, 0.75, 'UNIFORM\n+ HIGH CONF', ha='center', va='center',
            fontsize=9, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
    ax1.text(0.35, 0.55, 'SELECTIVE\n+ LOW CONF', ha='center', va='center',
            fontsize=9, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    ax1.text(0.85, 0.55, 'UNIFORM\n+ LOW CONF', ha='center', va='center',
            fontsize=9, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

    # Mark promoted sites from edge case analysis
    edge_report = report.get('edge_case_analysis', {})
    improvements = edge_report.get('confidence_improvements', [])

    for imp in improvements:
        site_id = imp['site_id']
        if site_id < len(sites):
            site = sites[site_id]
            ent = site.get('wavelength_entropy', 0.5)
            ax1.annotate(f"Site {site_id}\n+{imp['boost_breakdown']['total_boost']:.2f}",
                        (ent, imp['enhanced_confidence']),
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=8, fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

    ax1.set_xlabel('Wavelength Entropy (lower = more selective)', fontsize=11)
    ax1.set_ylabel('Overall Confidence', fontsize=11)
    ax1.set_title('Entropy vs Confidence Analysis', fontsize=12)
    ax1.set_xlim(0, 1.05)
    ax1.set_ylim(0.4, 0.85)
    ax1.grid(True, alpha=0.3)

    # Legend
    handles = [mpatches.Patch(color=c, label=l) for l, c in CONFIDENCE_COLORS.items()]
    ax1.legend(handles=handles, loc='lower left')

    # Right plot: Boost breakdown for enhanced sites
    ax2 = axes[1]

    if improvements:
        site_ids = [f"Site {imp['site_id']}" for imp in improvements[:10]]
        chromophore_boosts = [imp['boost_breakdown']['chromophore_boost'] for imp in improvements[:10]]
        burst_boosts = [imp['boost_breakdown']['burst_boost'] for imp in improvements[:10]]
        selectivity_boosts = [imp['boost_breakdown']['selectivity_boost'] for imp in improvements[:10]]

        x = np.arange(len(site_ids))
        width = 0.25

        ax2.bar(x - width, chromophore_boosts, width, label='Chromophore', color='#3498db')
        ax2.bar(x, burst_boosts, width, label='Burst', color='#e74c3c')
        ax2.bar(x + width, selectivity_boosts, width, label='Selectivity', color='#2ecc71')

        ax2.set_xlabel('Site', fontsize=11)
        ax2.set_ylabel('Confidence Boost', fontsize=11)
        ax2.set_title('Boost Breakdown for Enhanced Sites', fontsize=12)
        ax2.set_xticks(x)
        ax2.set_xticklabels(site_ids, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
    else:
        ax2.text(0.5, 0.5, 'No sites enhanced\n(Edge case not triggered)',
                ha='center', va='center', fontsize=12, transform=ax2.transAxes)
        ax2.set_title('Boost Breakdown (N/A)', fontsize=12)

    plt.tight_layout()
    output_path = Path(output_dir) / 'Figure12_confidence_enhancement.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def figure_13_chemical_heatmap(sites, report, output_dir):
    """
    Figure 13: Chemical Environment Heatmap

    Residue-level chromophore exposure map showing:
    - Which residues are most active
    - Wavelength-specific intensity distributions
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    # Collect residue-level data
    residue_wavelengths = defaultdict(lambda: defaultdict(int))
    residue_total_spikes = defaultdict(int)

    for site in sites:
        if site['category'] not in ['HIGH', 'MEDIUM']:
            continue
        if 'tier2' not in site or not site['tier2']:
            continue

        for res_str, contrib in site['tier2']['residue_contributions'].items():
            res_id = int(res_str)
            residue_total_spikes[res_id] += contrib['total_spikes']
            for wl, count in contrib.get('spikes_per_wavelength', {}).items():
                residue_wavelengths[res_id][wl] += count

    if not residue_wavelengths:
        print("  Warning: No residue data found for heatmap")
        plt.close()
        return

    # Sort residues by total spikes and take top 30
    sorted_residues = sorted(residue_total_spikes.items(), key=lambda x: x[1], reverse=True)[:30]
    top_residues = [r[0] for r in sorted_residues]

    # Build heatmap matrix
    wavelengths = ['250', '258', '265', '274', '280', '290']
    matrix = np.zeros((len(top_residues), len(wavelengths)))

    for i, res_id in enumerate(top_residues):
        total = residue_total_spikes[res_id]
        for j, wl in enumerate(wavelengths):
            count = residue_wavelengths[res_id].get(wl, 0)
            matrix[i, j] = 100 * count / total if total > 0 else 0

    # Left plot: Heatmap
    ax1 = axes[0]
    im = ax1.imshow(matrix, aspect='auto', cmap='YlOrRd')

    ax1.set_xticks(range(len(wavelengths)))
    ax1.set_xticklabels([CHROMOPHORE_LABELS[wl] for wl in wavelengths], rotation=45, ha='right')
    ax1.set_yticks(range(len(top_residues)))
    ax1.set_yticklabels([f"Res {r}" for r in top_residues])

    ax1.set_xlabel('Wavelength Channel', fontsize=11)
    ax1.set_ylabel('Residue (sorted by spike count)', fontsize=11)
    ax1.set_title('Residue Chromophore Exposure Map\n(% spikes per wavelength)', fontsize=12)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('% of Spikes', fontsize=10)

    # Add text annotations for high values
    for i in range(len(top_residues)):
        for j in range(len(wavelengths)):
            val = matrix[i, j]
            if val > 30:
                color = 'white' if val > 50 else 'black'
                ax1.text(j, i, f'{val:.0f}', ha='center', va='center',
                        fontsize=7, color=color, fontweight='bold')

    # Right plot: Total spike distribution by residue
    ax2 = axes[1]

    residue_labels = [f"Res {r}" for r in top_residues]
    totals = [residue_total_spikes[r] for r in top_residues]

    # Color bars by dominant wavelength
    bar_colors = []
    for res_id in top_residues:
        wl_counts = residue_wavelengths[res_id]
        if wl_counts:
            dom_wl = max(wl_counts.items(), key=lambda x: x[1])[0]
            bar_colors.append(WAVELENGTH_COLORS.get(dom_wl, '#95a5a6'))
        else:
            bar_colors.append('#95a5a6')

    bars = ax2.barh(range(len(top_residues)), totals, color=bar_colors, edgecolor='black', linewidth=0.5)

    ax2.set_yticks(range(len(top_residues)))
    ax2.set_yticklabels(residue_labels)
    ax2.set_xlabel('Total Spikes', fontsize=11)
    ax2.set_ylabel('Residue', fontsize=11)
    ax2.set_title('Residue Activity Ranking\n(color = dominant wavelength)', fontsize=12)
    ax2.invert_yaxis()  # Highest at top
    ax2.grid(True, alpha=0.3, axis='x')

    # Legend for wavelength colors
    handles = [mpatches.Patch(color=c, label=CHROMOPHORE_LABELS[wl])
               for wl, c in WAVELENGTH_COLORS.items()]
    ax2.legend(handles=handles, loc='lower right', fontsize=8)

    plt.tight_layout()
    output_path = Path(output_dir) / 'Figure13_chemical_heatmap.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def figure_selectivity_distribution(sites, report, output_dir):
    """
    Bonus Figure: Wavelength Selectivity Distribution

    Shows the distribution of wavelength entropy across all sites,
    with markers for different confidence levels.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Entropy histogram
    ax1 = axes[0]

    high_entropy = [s['wavelength_entropy'] for s in sites if s['category'] == 'HIGH']
    med_entropy = [s['wavelength_entropy'] for s in sites if s['category'] == 'MEDIUM']
    low_entropy = [s['wavelength_entropy'] for s in sites if s['category'] == 'LOW']

    bins = np.linspace(0, 1, 21)
    ax1.hist(high_entropy, bins=bins, alpha=0.7, color=CONFIDENCE_COLORS['HIGH'],
             label=f'HIGH ({len(high_entropy)})', edgecolor='black', linewidth=0.5)
    ax1.hist(med_entropy, bins=bins, alpha=0.5, color=CONFIDENCE_COLORS['MEDIUM'],
             label=f'MEDIUM ({len(med_entropy)})', edgecolor='black', linewidth=0.5)

    ax1.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='High selectivity threshold')
    ax1.axvline(x=0.8, color='orange', linestyle='--', alpha=0.7, label='Low selectivity threshold')

    ax1.set_xlabel('Wavelength Entropy', fontsize=11)
    ax1.set_ylabel('Number of Sites', fontsize=11)
    ax1.set_title('Wavelength Selectivity Distribution', fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Right: Dominant wavelength pie chart
    ax2 = axes[1]

    chrom = report['chromophore_selectivity']['dominant_chromophores']
    labels = ['TRP\n(280nm)', 'TYR\n(274nm)', 'PHE\n(258nm)', 'S-S\n(250nm)', 'Thermal\n(290nm)', 'Other']
    sizes = [
        chrom['trp_280nm_sites'],
        chrom['tyr_274nm_sites'],
        chrom['phe_258nm_sites'],
        chrom['ss_250nm_sites'],
        chrom['thermal_290nm_sites'],
        chrom['other_sites'],
    ]
    colors = [WAVELENGTH_COLORS['280'], WAVELENGTH_COLORS['274'], WAVELENGTH_COLORS['258'],
              WAVELENGTH_COLORS['250'], WAVELENGTH_COLORS['290'], '#95a5a6']

    # Only show non-zero slices
    filtered = [(l, s, c) for l, s, c in zip(labels, sizes, colors) if s > 0]
    if filtered:
        labels, sizes, colors = zip(*filtered)
        wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                           startangle=90, explode=[0.02]*len(sizes))
        ax2.set_title('Sites by Dominant Chromophore', fontsize=12)

    plt.tight_layout()
    output_path = Path(output_dir) / 'Figure_selectivity_distribution.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def figure_performance_summary(report, output_dir):
    """
    Bonus Figure: Performance & Quality Summary

    Visual dashboard of key performance metrics.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    summary = report['summary']
    perf = report['performance_qc']

    # Top-left: Detection metrics
    ax1 = axes[0, 0]
    metrics = ['Total Spikes', 'Sites Found', 'HIGH Conf', 'MEDIUM Conf']
    values = [summary['total_spikes'], summary['sites_found'],
              summary['high_confidence'], summary['medium_confidence']]
    colors = ['#3498db', '#9b59b6', '#27ae60', '#f39c12']

    bars = ax1.bar(metrics, values, color=colors, edgecolor='black', linewidth=1)
    ax1.set_ylabel('Count', fontsize=11)
    ax1.set_title('Detection Summary', fontsize=12)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.02,
                f'{val:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Top-right: Signal quality pie
    ax2 = axes[0, 1]
    dq = perf['detection_quality']

    aromatic = dq['aromatic_signal_ratio'] * 100
    thermal = dq['thermal_noise_ratio'] * 100
    other = 100 - aromatic - thermal

    sizes = [aromatic, thermal, other]
    labels = [f'Aromatic Signal\n({aromatic:.1f}%)',
              f'Thermal Noise\n({thermal:.1f}%)',
              f'Other\n({other:.1f}%)']
    colors = ['#27ae60', '#e74c3c', '#95a5a6']

    ax2.pie(sizes, labels=labels, colors=colors, autopct='', startangle=90,
            explode=[0.02, 0.05, 0])
    ax2.set_title('Signal Quality Distribution', fontsize=12)

    # Bottom-left: Quality indicators
    ax3 = axes[1, 0]
    indicators = ['High Entropy\n+ High Conf', 'Single Residue\nSites', 'Low Spike\n+ High Conf']
    values = [dq['high_entropy_high_confidence'], dq['single_residue_sites'],
              dq['low_spike_high_confidence']]
    colors = ['#e74c3c' if v > 5 else '#f39c12' if v > 0 else '#27ae60' for v in values]

    bars = ax3.bar(indicators, values, color=colors, edgecolor='black', linewidth=1)
    ax3.set_ylabel('Count (potential false positives)', fontsize=10)
    ax3.set_title('Quality Control Indicators', fontsize=12)
    ax3.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Bottom-right: Performance metrics
    ax4 = axes[1, 1]
    ce = perf['computational_efficiency']

    perf_text = f"""
    COMPUTATIONAL EFFICIENCY

    Total Time: {ce['total_time_ms']:,} ms ({ce['total_time_ms']/1000:.2f} s)

    Throughput:
      • Frames/second: {ce['frames_per_second']:.1f}
      • Sites/second: {ce['sites_per_second']:.1f}

    Analysis:
      • {summary['frames_analyzed']} frames analyzed
      • {summary['sites_found']} sites detected
      • {summary['total_spikes']:,} total spikes
    """

    ax4.text(0.1, 0.5, perf_text, transform=ax4.transAxes, fontsize=11,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    ax4.axis('off')
    ax4.set_title('Performance Summary', fontsize=12)

    plt.suptitle(f"PRISM4D Analysis Report - {report['pdb_id']}", fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = Path(output_dir) / 'Figure_performance_summary.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


###############################################################################
# PYMOL PUBLICATION-QUALITY VISUALIZATION AND MOVIE GENERATION
###############################################################################

# PyMOL color definitions matching our wavelength scheme
PYMOL_WAVELENGTH_COLORS = {
    '250': ('ss_red', '0xe74c3c'),        # S-S Disulfide - Red
    '258': ('phe_orange', '0xe67e22'),     # PHE - Orange
    '265': ('general_yellow', '0xf1c40f'), # General - Yellow
    '274': ('tyr_green', '0x2ecc71'),      # TYR - Green
    '280': ('trp_blue', '0x3498db'),       # TRP - Blue
    '290': ('thermal_purple', '0x9b59b6'), # Thermal - Purple
}

PYMOL_CONFIDENCE_COLORS = {
    'HIGH': ('conf_high', '0x27ae60'),      # Green
    'MEDIUM': ('conf_medium', '0xf39c12'),  # Orange
    'LOW': ('conf_low', '0x95a5a6'),        # Gray
}


def generate_pymol_master_session(sites, report, output_dir):
    """
    Generate publication-quality PyMOL session script.

    Creates a comprehensive visualization with:
    - Structure overview with cryptic sites
    - Confidence-tiered coloring
    - Wavelength-selective views
    - Surface pocket analysis
    - Interface highlighting for complexes
    """
    output_dir = Path(output_dir)
    pdb_id = report['pdb_id']

    # Sort sites by confidence
    high_sites = [s for s in sites if s['category'] == 'HIGH']
    medium_sites = [s for s in sites if s['category'] == 'MEDIUM']
    low_sites = [s for s in sites if s['category'] == 'LOW']

    # Get top 20 sites for detailed visualization
    all_by_conf = sorted(sites, key=lambda x: x.get('overall_confidence', 0), reverse=True)
    top_sites = all_by_conf[:20]

    pml_content = f'''# ============================================================================
# PRISM4D Publication-Quality PyMOL Session
# PDB: {pdb_id}
# Generated: {report.get('analysis_timestamp', 'N/A')}
# ============================================================================
#
# This script generates pharma-actionable visualizations for:
#   - Drug discovery presentations
#   - Publication figures
#   - Grant applications
#   - Regulatory submissions
#
# Usage:
#   pymol -qc {pdb_id}_PRISM4D_master.pml
#   OR: File > Run Script in PyMOL GUI
#
# Movies are saved to: {pdb_id}_movies/
# ============================================================================

# -----------------------------------------------------------------------------
# SETUP: Environment and Colors
# -----------------------------------------------------------------------------

# Fetch structure
fetch {pdb_id}, async=0
remove solvent
remove resn HOH

# Create working objects
create protein, {pdb_id} and polymer
create ligands, {pdb_id} and organic
create glycans, {pdb_id} and (resn NAG or resn MAN or resn BMA or resn FUC or resn GAL)

# Define custom colors for wavelength selectivity
'''

    # Add color definitions
    for wl, (name, hex_color) in PYMOL_WAVELENGTH_COLORS.items():
        r = int(hex_color[2:4], 16) / 255
        g = int(hex_color[4:6], 16) / 255
        b = int(hex_color[6:8], 16) / 255
        pml_content += f'set_color {name}, [{r:.3f}, {g:.3f}, {b:.3f}]\n'

    for conf, (name, hex_color) in PYMOL_CONFIDENCE_COLORS.items():
        r = int(hex_color[2:4], 16) / 255
        g = int(hex_color[4:6], 16) / 255
        b = int(hex_color[6:8], 16) / 255
        pml_content += f'set_color {name}, [{r:.3f}, {g:.3f}, {b:.3f}]\n'

    pml_content += '''
# Gradient colors for confidence
set_color conf_gradient_1, [0.15, 0.68, 0.38]  # High
set_color conf_gradient_2, [0.95, 0.61, 0.07]  # Medium
set_color conf_gradient_3, [0.58, 0.65, 0.65]  # Low

# Publication-quality settings
bg_color white
set ray_shadows, 0
set ray_trace_mode, 1
set antialias, 2
set ambient, 0.4
set reflect, 0.5
set direct, 0.7
set specular, 0.5
set shininess, 40
set fog, 0
set depth_cue, 0
set ray_opaque_background, 1
set orthoscopic, 1

# Label settings
set label_size, 16
set label_color, black
set label_font_id, 7
set label_outline_color, white

# -----------------------------------------------------------------------------
# SCENE 1: Structure Overview with All Cryptic Sites
# -----------------------------------------------------------------------------

hide everything
show cartoon, protein
color gray80, protein

# Color chains differently for complexes
'''

    # Detect chains and color them
    pml_content += '''
python
# Color each chain distinctly
cmd.do("select chain_A, chain A")
cmd.do("select chain_B, chain B")
cmd.do("select chain_C, chain C")
cmd.do("select chain_E, chain E")

chains = cmd.get_chains("protein")
chain_colors = ["lightblue", "lightorange", "palegreen", "lightpink", "paleyellow", "palecyan"]
for i, chain in enumerate(chains):
    if i < len(chain_colors):
        cmd.color(chain_colors[i], f"chain {chain} and protein")
python end

'''

    # Add cryptic site pseudoatoms
    pml_content += '''
# -----------------------------------------------------------------------------
# CRYPTIC SITES: Pseudoatoms with confidence-based sizing
# -----------------------------------------------------------------------------

'''

    for i, site in enumerate(top_sites):
        site_id = i + 1
        pos = site.get('centroid', [0, 0, 0])
        conf = site.get('overall_confidence', 0.5)
        spikes = site.get('spike_count', 0)
        category = site.get('category', 'LOW')

        # Get dominant wavelength for coloring
        dom_wl = site.get('dominant_wavelength')
        if dom_wl:
            wl_key = str(int(round(dom_wl)))
            if wl_key in PYMOL_WAVELENGTH_COLORS:
                site_color = PYMOL_WAVELENGTH_COLORS[wl_key][0]
            else:
                site_color = 'gray50'
        else:
            site_color = 'gray50'

        # Sphere scale based on burst intensity
        max_burst = site.get('max_single_frame_spikes', 50)
        sphere_scale = 1.5 + min(max_burst / 100, 2.0)  # 1.5 to 3.5 range

        # Transparency based on confidence
        transparency = max(0, 0.5 - conf * 0.5)

        pml_content += f'''# Site {site_id}: {category} confidence ({conf:.2f}), {spikes} spikes
pseudoatom site_{site_id}, pos=[{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]
show spheres, site_{site_id}
color {site_color}, site_{site_id}
set sphere_scale, {sphere_scale:.2f}, site_{site_id}
set sphere_transparency, {transparency:.2f}, site_{site_id}
'''

        # Add residue selections
        residues = site.get('residues', [])[:10]  # Top 10 contributing residues
        if residues:
            resi_str = ' or resi '.join([str(r) for r in residues])
            pml_content += f'select site_{site_id}_residues, resi {resi_str}\n'

    # Group sites by confidence
    high_ids = [str(i+1) for i, s in enumerate(top_sites) if s['category'] == 'HIGH']
    med_ids = [str(i+1) for i, s in enumerate(top_sites) if s['category'] == 'MEDIUM']
    low_ids = [str(i+1) for i, s in enumerate(top_sites) if s['category'] == 'LOW']

    if high_ids:
        pml_content += f'\ngroup high_confidence_sites, site_{" or site_".join(high_ids)}\n'
    if med_ids:
        pml_content += f'group medium_confidence_sites, site_{" or site_".join(med_ids)}\n'
    if low_ids:
        pml_content += f'group low_confidence_sites, site_{" or site_".join(low_ids)}\n'

    pml_content += '\ngroup all_cryptic_sites, site_*\n'

    # Add wavelength-selective views
    pml_content += '''
# -----------------------------------------------------------------------------
# SCENE 2: Wavelength-Selective Views
# -----------------------------------------------------------------------------

# S-S Disulfide sites (250nm) - Drug design hotspots for covalent inhibitors
'''

    ss_sites = [s for s in sites if s.get('dominant_wavelength') and abs(s['dominant_wavelength'] - 250) < 10]
    if ss_sites:
        pml_content += f'# Found {len(ss_sites)} S-S selective sites\n'
        for i, site in enumerate(ss_sites[:5]):
            pos = site.get('centroid', [0, 0, 0])
            pml_content += f'pseudoatom ss_site_{i+1}, pos=[{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]\n'
            pml_content += f'color ss_red, ss_site_{i+1}\n'
            pml_content += f'show spheres, ss_site_{i+1}\n'
        pml_content += 'group ss_disulfide_sites, ss_site_*\n'
        pml_content += 'show sticks, resn CYS\n'
        pml_content += 'color yellow, resn CYS\n'

    # TRP sites (280nm)
    trp_sites = [s for s in sites if s.get('dominant_wavelength') and abs(s['dominant_wavelength'] - 280) < 5]
    if trp_sites:
        pml_content += f'\n# Tryptophan-selective sites (280nm) - {len(trp_sites)} sites\n'
        for i, site in enumerate(trp_sites[:5]):
            pos = site.get('centroid', [0, 0, 0])
            pml_content += f'pseudoatom trp_site_{i+1}, pos=[{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]\n'
            pml_content += f'color trp_blue, trp_site_{i+1}\n'
            pml_content += f'show spheres, trp_site_{i+1}\n'
        pml_content += 'group trp_sites, trp_site_*\n'
        pml_content += 'show sticks, resn TRP\n'
        pml_content += 'color marine, resn TRP\n'

    # Add surface pocket analysis
    pml_content += '''
# -----------------------------------------------------------------------------
# SCENE 3: Surface Pocket Analysis
# -----------------------------------------------------------------------------

# Create surface for pocket visualization
create protein_surface, protein
show surface, protein_surface
set surface_color, gray90, protein_surface
set transparency, 0.7, protein_surface

# Clip surface around cryptic sites to reveal pockets
# (Adjust clip distance as needed)

'''

    # Add interface analysis for multi-chain complexes
    pml_content += '''
# -----------------------------------------------------------------------------
# SCENE 4: Interface Analysis (for complexes)
# -----------------------------------------------------------------------------

python
chains = cmd.get_chains("protein")
if len(chains) >= 2:
    # Define interface residues (within 4A of other chain)
    for i, chain1 in enumerate(chains):
        for chain2 in chains[i+1:]:
            cmd.select(f"interface_{chain1}_{chain2}",
                      f"(chain {chain1} within 4 of chain {chain2}) or (chain {chain2} within 4 of chain {chain1})")
            cmd.show("sticks", f"interface_{chain1}_{chain2}")
            cmd.color("salmon", f"interface_{chain1}_{chain2}")
    cmd.group("interfaces", "interface_*")
python end

'''

    # Store scenes for easy navigation
    pml_content += '''
# -----------------------------------------------------------------------------
# SAVED SCENES (F1-F8 keys)
# -----------------------------------------------------------------------------

# Scene 1: Overview with all sites
zoom all_cryptic_sites, 10
scene F1, store, Overview - All Cryptic Sites

# Scene 2: High confidence sites only
hide spheres, medium_confidence_sites low_confidence_sites
zoom high_confidence_sites, 8
scene F2, store, High Confidence Sites Only

# Scene 3: Surface pocket view
show everything
hide spheres
show surface, protein_surface
zoom all_cryptic_sites
scene F3, store, Surface Pocket Analysis

# Scene 4: Wavelength-selective view
hide surface
show spheres, ss_disulfide_sites trp_sites
zoom ss_disulfide_sites trp_sites, 10
scene F4, store, Wavelength-Selective Sites

# Reset to overview
scene F1

'''

    # Write the master script
    master_path = output_dir / f'{pdb_id}_PRISM4D_master.pml'
    with open(master_path, 'w') as f:
        f.write(pml_content)

    print(f"  Saved: {master_path}")
    return master_path


def generate_pymol_movie_scripts(sites, report, output_dir):
    """
    Generate PyMOL movie scripts for:
    - 360° rotation of structure with cryptic sites
    - Zoom tour through high-confidence sites
    - Wavelength channel comparison
    - Surface transparency reveal animation
    """
    output_dir = Path(output_dir)
    pdb_id = report['pdb_id']

    # Create movies directory
    movies_dir = output_dir / f'{pdb_id}_movies'
    movies_dir.mkdir(exist_ok=True)

    # Get top sites
    high_sites = [s for s in sites if s['category'] == 'HIGH']
    top_sites = sorted(sites, key=lambda x: x.get('overall_confidence', 0), reverse=True)[:10]

    # Movie 1: 360° Rotation
    rotation_script = f'''# ============================================================================
# PRISM4D Movie 1: 360° Rotation Showcase
# PDB: {pdb_id}
# ============================================================================
#
# Creates a smooth 360° rotation showing all cryptic sites
# Output: {pdb_id}_rotation.mp4 (or .mpg)
#
# Duration: 10 seconds at 30fps = 300 frames
# Resolution: 1920x1080 (Full HD)
# ============================================================================

# Load the master session first
@{pdb_id}_PRISM4D_master.pml

# Movie settings
set ray_trace_frames, 1
set cache_frames, 0

# Set up camera
reset
zoom all_cryptic_sites, 15
orient

# Configure movie
mset 1 x300

# Rotation: 360 degrees over 300 frames
python
for frame in range(1, 301):
    angle = (frame / 300) * 360
    cmd.frame(frame)
    cmd.rotate("y", 1.2)  # 360/300 = 1.2 degrees per frame
python end

# Set viewport for HD
viewport 1920, 1080

# Render movie
set ray_shadows, 0
set antialias, 2
movie.produce {movies_dir}/{pdb_id}_rotation.mp4, quality=90

# Alternative: Save as frames for external encoding
# mpng {movies_dir}/rotation_frames/frame_, width=1920, height=1080
'''

    rotation_path = output_dir / f'{pdb_id}_movie_rotation.pml'
    with open(rotation_path, 'w') as f:
        f.write(rotation_script)
    print(f"  Saved: {rotation_path}")

    # Movie 2: Site Zoom Tour
    if top_sites:
        tour_script = f'''# ============================================================================
# PRISM4D Movie 2: Cryptic Site Zoom Tour
# PDB: {pdb_id}
# ============================================================================
#
# Visits each high-confidence cryptic site with smooth transitions
# Output: {pdb_id}_site_tour.mp4
#
# Duration: ~{len(top_sites) * 3} seconds (3 sec per site)
# ============================================================================

# Load master session
@{pdb_id}_PRISM4D_master.pml

# Movie settings
set ray_trace_frames, 1

# Calculate frames: 3 seconds per site at 30fps = 90 frames per site
# Split: 30 frames zoom in, 30 frames hold, 30 frames transition
'''
        total_frames = len(top_sites) * 90
        tour_script += f'\nmset 1 x{total_frames}\n\n'

        for i, site in enumerate(top_sites):
            pos = site.get('centroid', [0, 0, 0])
            start_frame = i * 90 + 1

            tour_script += f'''# Site {i+1}: {site.get('category', 'N/A')} confidence
# Frames {start_frame}-{start_frame + 89}
frame {start_frame}
zoom site_{i+1}, 8, {start_frame + 29}, animate=1
mdo {start_frame + 30}: show sticks, site_{i+1}_residues
mdo {start_frame + 60}: hide sticks, site_{i+1}_residues

'''

        tour_script += f'''
# Render
viewport 1920, 1080
movie.produce {movies_dir}/{pdb_id}_site_tour.mp4, quality=90
'''

        tour_path = output_dir / f'{pdb_id}_movie_site_tour.pml'
        with open(tour_path, 'w') as f:
            f.write(tour_script)
        print(f"  Saved: {tour_path}")

    # Movie 3: Surface Transparency Reveal
    reveal_script = f'''# ============================================================================
# PRISM4D Movie 3: Surface Transparency Reveal
# PDB: {pdb_id}
# ============================================================================
#
# Animates surface transparency to reveal buried cryptic sites
# Output: {pdb_id}_surface_reveal.mp4
#
# Duration: 6 seconds (180 frames)
# ============================================================================

# Load master session
@{pdb_id}_PRISM4D_master.pml

# Setup
hide everything
show cartoon, protein
show spheres, all_cryptic_sites
create surf, protein
show surface, surf
set surface_color, white, surf

# Movie: Fade surface from opaque to transparent
mset 1 x180

python
# Frames 1-60: Opaque surface
for frame in range(1, 61):
    cmd.mdo(frame, f"set transparency, 0, surf")

# Frames 61-120: Fade out surface
for frame in range(61, 121):
    trans = (frame - 60) / 60.0 * 0.85
    cmd.mdo(frame, f"set transparency, {{trans:.3f}}, surf")

# Frames 121-180: Hold transparent, slight rotation
for frame in range(121, 181):
    cmd.mdo(frame, "rotate y, 0.5")
python end

# Render
viewport 1920, 1080
movie.produce {movies_dir}/{pdb_id}_surface_reveal.mp4, quality=90
'''

    reveal_path = output_dir / f'{pdb_id}_movie_surface_reveal.pml'
    with open(reveal_path, 'w') as f:
        f.write(reveal_script)
    print(f"  Saved: {reveal_path}")

    # Movie 4: Wavelength Channel Comparison
    channel_script = f'''# ============================================================================
# PRISM4D Movie 4: Wavelength Channel Comparison
# PDB: {pdb_id}
# ============================================================================
#
# Shows sites appearing by wavelength selectivity
# S-S (250nm) → PHE (258nm) → TYR (274nm) → TRP (280nm) → Thermal (290nm)
#
# Output: {pdb_id}_wavelength_channels.mp4
# Duration: 10 seconds (300 frames)
# ============================================================================

# Load master session
@{pdb_id}_PRISM4D_master.pml

# Hide all site spheres initially
hide spheres

# Setup movie
mset 1 x300

# Each channel gets ~60 frames (2 seconds)
python
# Frame 1-60: S-S (250nm) - Red
for f in range(1, 61):
    cmd.mdo(f, "show spheres, ss_disulfide_sites; color ss_red, ss_disulfide_sites")

# Frame 61-120: Add PHE (258nm) - Orange
for f in range(61, 121):
    cmd.mdo(f, "show spheres, site_* and name PS*; color phe_orange, (all within 5 of resn PHE)")

# Frame 121-180: Add TYR (274nm) - Green
for f in range(121, 181):
    cmd.mdo(f, "color tyr_green, (all within 5 of resn TYR)")

# Frame 181-240: Add TRP (280nm) - Blue
for f in range(181, 241):
    cmd.mdo(f, "show spheres, trp_sites; color trp_blue, trp_sites")

# Frame 241-300: Show all + gentle rotation
for f in range(241, 301):
    cmd.mdo(f, "show spheres, all_cryptic_sites; rotate y, 0.6")
python end

# Render
viewport 1920, 1080
movie.produce {movies_dir}/{pdb_id}_wavelength_channels.mp4, quality=90
'''

    channel_path = output_dir / f'{pdb_id}_movie_wavelength_channels.pml'
    with open(channel_path, 'w') as f:
        f.write(channel_script)
    print(f"  Saved: {channel_path}")

    # Create a runner script
    runner_script = f'''#!/bin/bash
# ============================================================================
# PRISM4D Movie Generation Runner
# PDB: {pdb_id}
# ============================================================================
#
# Generates all publication-quality movies
# Requires: PyMOL with movie support
#
# Usage: bash {pdb_id}_generate_movies.sh
# ============================================================================

set -e

echo "PRISM4D Movie Generator for {pdb_id}"
echo "========================================"

# Create movies directory
mkdir -p {movies_dir}

# Check PyMOL
if ! command -v pymol &> /dev/null; then
    echo "ERROR: PyMOL not found. Please install PyMOL."
    exit 1
fi

echo ""
echo "[1/4] Generating 360° rotation movie..."
pymol -qc {pdb_id}_movie_rotation.pml

echo ""
echo "[2/4] Generating site tour movie..."
pymol -qc {pdb_id}_movie_site_tour.pml

echo ""
echo "[3/4] Generating surface reveal movie..."
pymol -qc {pdb_id}_movie_surface_reveal.pml

echo ""
echo "[4/4] Generating wavelength channels movie..."
pymol -qc {pdb_id}_movie_wavelength_channels.pml

echo ""
echo "========================================"
echo "All movies generated in: {movies_dir}/"
echo ""
ls -la {movies_dir}/
'''

    runner_path = output_dir / f'{pdb_id}_generate_movies.sh'
    with open(runner_path, 'w') as f:
        f.write(runner_script)
    os.chmod(runner_path, 0o755)
    print(f"  Saved: {runner_path}")

    return movies_dir


def generate_pymol_pharma_actionable(sites, report, output_dir):
    """
    Generate pharma-specific actionable PyMOL outputs:
    - Druggable pocket highlighting
    - Covalent inhibitor targets (cysteine sites)
    - Allosteric site candidates
    - Interface disruption opportunities
    """
    output_dir = Path(output_dir)
    pdb_id = report['pdb_id']

    # Identify pharma-relevant sites
    ss_sites = [s for s in sites if s.get('dominant_wavelength') and abs(s['dominant_wavelength'] - 250) < 10]
    high_conf = [s for s in sites if s['category'] == 'HIGH']

    pml_content = f'''# ============================================================================
# PRISM4D Pharma-Actionable Analysis
# PDB: {pdb_id}
# ============================================================================
#
# Highlights drug discovery opportunities:
#   1. Covalent inhibitor targets (S-S/CYS sites)
#   2. High-confidence druggable pockets
#   3. Allosteric site candidates
#   4. Interface disruption opportunities
#
# For: Medicinal chemistry, hit-to-lead optimization
# ============================================================================

# Load master session
@{pdb_id}_PRISM4D_master.pml

# -----------------------------------------------------------------------------
# COVALENT INHIBITOR TARGETS
# -----------------------------------------------------------------------------
# Sites near cysteine residues are candidates for covalent warheads
# (acrylamides, chloroacetamides, etc.)

'''

    if ss_sites:
        pml_content += f'# Found {len(ss_sites)} S-S selective sites (250nm)\n'
        pml_content += '''
# Highlight all cysteines
select all_cys, resn CYS
show sticks, all_cys
color yellow, all_cys

# Show cysteine sulfurs prominently
select cys_sulfurs, resn CYS and name SG
show spheres, cys_sulfurs
set sphere_scale, 0.4, cys_sulfurs
color orange, cys_sulfurs

# Label with residue numbers
label cys_sulfurs, "%s%s" % (resn, resi)

# Create selection for reactive cysteines (solvent accessible)
'''
        for i, site in enumerate(ss_sites[:5]):
            residues = site.get('residues', [])
            cys_residues = [r for r in residues if r in [int(x) for x in str(residues).split() if x.isdigit()]]
            if residues:
                pml_content += f'\n# S-S Site {i+1}: Potential covalent target\n'
                pml_content += f'select cov_target_{i+1}, resi {" or resi ".join(str(r) for r in residues[:5])}\n'
                pml_content += f'show sticks, cov_target_{i+1}\n'
                pml_content += f'color salmon, cov_target_{i+1}\n'

    pml_content += '''
# -----------------------------------------------------------------------------
# ALLOSTERIC SITE CANDIDATES
# -----------------------------------------------------------------------------
# High-confidence cryptic sites distant from active site

# Calculate distances from active site center (if known)
# Sites >15A from active site are allosteric candidates

'''

    if high_conf:
        pml_content += f'# Found {len(high_conf)} high-confidence sites\n'
        for i, site in enumerate(high_conf[:5]):
            pos = site.get('centroid', [0, 0, 0])
            conf = site.get('overall_confidence', 0)
            pml_content += f'''
# Allosteric candidate {i+1}: confidence {conf:.2f}
pseudoatom allosteric_{i+1}, pos=[{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]
show spheres, allosteric_{i+1}
color splitpea, allosteric_{i+1}
set sphere_scale, 2.5, allosteric_{i+1}
'''
        pml_content += 'group allosteric_candidates, allosteric_*\n'

    pml_content += '''
# -----------------------------------------------------------------------------
# INTERFACE DISRUPTION OPPORTUNITIES
# -----------------------------------------------------------------------------
# Cryptic sites at protein-protein interfaces

python
chains = cmd.get_chains("protein")
if len(chains) >= 2:
    print(f"Multi-chain complex detected: {chains}")
    print("Checking for cryptic sites at interfaces...")

    # Interface definition
    cmd.select("interface_all", "(chain A within 5 of chain B) or (chain B within 5 of chain A)")
    cmd.show("surface", "interface_all")
    cmd.set("surface_color", "palegreen", "interface_all")
    cmd.set("transparency", 0.5, "interface_all")
python end

# -----------------------------------------------------------------------------
# DRUGGABILITY SCORE VISUALIZATION
# -----------------------------------------------------------------------------
# Color sites by druggability (based on confidence + burst intensity)

python
# Get all site pseudoatoms
sites = [name for name in cmd.get_names() if name.startswith("site_")]

for site_name in sites:
    # Extract site number
    try:
        site_num = int(site_name.split("_")[1])
    except:
        continue

    # Color gradient based on sphere scale (which encodes confidence)
    scale = cmd.get("sphere_scale", site_name)
    if scale > 3.0:
        cmd.color("green", site_name)  # Highly druggable
    elif scale > 2.5:
        cmd.color("yellow", site_name)  # Moderately druggable
    else:
        cmd.color("gray", site_name)  # Lower priority
python end

# -----------------------------------------------------------------------------
# SAVE PHARMA SESSION
# -----------------------------------------------------------------------------

# Store scene
zoom all_cryptic_sites, 15
scene pharma_overview, store, Pharma Actionable Overview

# Save session
save ''' + str(output_dir / f'{pdb_id}_pharma_actionable.pse') + '''

print "Pharma-actionable session saved!"
print "Key features:"
print "  - Covalent targets: cov_target_* selections"
print "  - Allosteric sites: allosteric_* pseudoatoms"
print "  - Interface opportunities: interface_all selection"
'''

    pharma_path = output_dir / f'{pdb_id}_pharma_actionable.pml'
    with open(pharma_path, 'w') as f:
        f.write(pml_content)
    print(f"  Saved: {pharma_path}")

    return pharma_path


def generate_pymol_figure_panels(sites, report, output_dir):
    """
    Generate ready-to-use figure panels for publications.
    Creates specific views optimized for:
    - Figure panel A: Structure overview
    - Figure panel B: Cryptic site closeup
    - Figure panel C: Wavelength comparison
    - Figure panel D: Surface analysis
    """
    output_dir = Path(output_dir)
    pdb_id = report['pdb_id']

    panel_script = f'''# ============================================================================
# PRISM4D Publication Figure Panels
# PDB: {pdb_id}
# ============================================================================
#
# Generates 4 standard figure panels at 300 DPI
# Output: {pdb_id}_panels/ directory
#
# Panel A: Structure overview (landscape)
# Panel B: Top cryptic site closeup (square)
# Panel C: Wavelength selectivity comparison (2x3 grid)
# Panel D: Surface pocket analysis (landscape)
# ============================================================================

# Load master session
@{pdb_id}_PRISM4D_master.pml

# Create output directory
python
import os
panels_dir = "{output_dir}/{pdb_id}_panels"
os.makedirs(panels_dir, exist_ok=True)
python end

# Publication quality settings
set ray_shadows, 0
set ray_trace_mode, 1
set antialias, 3
set hash_max, 300
set ray_trace_fog, 0

# -----------------------------------------------------------------------------
# PANEL A: Structure Overview (3:2 landscape, 300 DPI)
# -----------------------------------------------------------------------------

hide everything
show cartoon, protein
color gray80, protein
show spheres, all_cryptic_sites

# Color chains
spectrum chain, blue_white_red, protein

# Orient for best view
orient protein
turn y, 30
turn x, -15

# Render at 300 DPI (assume 6" x 4" = 1800 x 1200 pixels)
ray 1800, 1200
png {output_dir}/{pdb_id}_panels/Panel_A_overview.png, dpi=300

# -----------------------------------------------------------------------------
# PANEL B: Top Cryptic Site Closeup (1:1 square)
# -----------------------------------------------------------------------------

# Zoom to site 1 (highest confidence)
zoom site_1, 8
show sticks, site_1_residues
color atomic, site_1_residues and not elem C
color gray50, site_1_residues and elem C

# Add key labels
python
cmd.label("site_1", '"Site 1"')
python end

# Render square (4" x 4" = 1200 x 1200)
ray 1200, 1200
png {output_dir}/{pdb_id}_panels/Panel_B_closeup.png, dpi=300

hide sticks, site_1_residues
hide labels

# -----------------------------------------------------------------------------
# PANEL C: Wavelength Selectivity (6 mini panels)
# -----------------------------------------------------------------------------

# We'll create one combined image showing 6 views
# Use PyMOL grid mode or manual arrangement

set grid_mode, 1
set grid_slot, 1
zoom ss_disulfide_sites
set grid_slot, 2
zoom trp_sites
# ... etc

# For simplicity, render individual wavelength views
hide spheres
show spheres, ss_disulfide_sites
zoom ss_disulfide_sites, 10
ray 800, 800
png {output_dir}/{pdb_id}_panels/Panel_C_250nm_SS.png, dpi=300

hide spheres, ss_disulfide_sites
show spheres, trp_sites
zoom trp_sites, 10
ray 800, 800
png {output_dir}/{pdb_id}_panels/Panel_C_280nm_TRP.png, dpi=300

set grid_mode, 0

# -----------------------------------------------------------------------------
# PANEL D: Surface Pocket Analysis (landscape)
# -----------------------------------------------------------------------------

reset
hide everything
show cartoon, protein
color gray90, protein
show surface, protein_surface
set transparency, 0.6, protein_surface
show spheres, high_confidence_sites
set sphere_scale, 2.0, high_confidence_sites
color green, high_confidence_sites

orient
turn y, 45

ray 1800, 1200
png {output_dir}/{pdb_id}_panels/Panel_D_surface.png, dpi=300

# -----------------------------------------------------------------------------
# SUMMARY
# -----------------------------------------------------------------------------

python
import os
panels = os.listdir("{output_dir}/{pdb_id}_panels")
print(f"\\nGenerated {{len(panels)}} figure panels:")
for p in sorted(panels):
    print(f"  - {{p}}")
python end

print ""
print "Figure panels saved to: {output_dir}/{pdb_id}_panels/"
print "Ready for publication assembly in Illustrator/Inkscape"
'''

    panel_path = output_dir / f'{pdb_id}_figure_panels.pml'
    with open(panel_path, 'w') as f:
        f.write(panel_script)
    print(f"  Saved: {panel_path}")

    return panel_path


def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_comprehensive_figures.py <output_dir>")
        print("\nExpects these files in output_dir:")
        print("  - comprehensive_report.json")
        print("  - cryptic_sites.json")
        sys.exit(1)

    output_dir = sys.argv[1]

    if not os.path.exists(output_dir):
        print(f"Error: Directory not found: {output_dir}")
        sys.exit(1)

    print("=" * 60)
    print("PRISM4D Comprehensive Figure Generator")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    try:
        report, sites = load_data(output_dir)
        print(f"  Loaded {len(sites)} sites from {report['pdb_id']}")
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    # Generate figures
    print("\nGenerating figures...")

    print("\n[Figure 11] Burst Event Timeline")
    figure_11_burst_timeline(sites, report, output_dir)

    print("\n[Figure 12] Confidence Enhancement Visualization")
    figure_12_confidence_enhancement(sites, report, output_dir)

    print("\n[Figure 13] Chemical Environment Heatmap")
    figure_13_chemical_heatmap(sites, report, output_dir)

    print("\n[Bonus] Selectivity Distribution")
    figure_selectivity_distribution(sites, report, output_dir)

    print("\n[Bonus] Performance Summary")
    figure_performance_summary(report, output_dir)

    # =========================================================================
    # PYMOL PUBLICATION & MOVIE GENERATION
    # =========================================================================
    print("\n" + "=" * 60)
    print("PYMOL VISUALIZATION GENERATION")
    print("=" * 60)

    print("\n[PyMOL] Master Publication Session")
    generate_pymol_master_session(sites, report, output_dir)

    print("\n[PyMOL] Movie Scripts")
    generate_pymol_movie_scripts(sites, report, output_dir)

    print("\n[PyMOL] Pharma-Actionable Analysis")
    generate_pymol_pharma_actionable(sites, report, output_dir)

    print("\n[PyMOL] Publication Figure Panels")
    generate_pymol_figure_panels(sites, report, output_dir)

    print("\n" + "=" * 60)
    print("ALL OUTPUTS GENERATED SUCCESSFULLY!")
    print("=" * 60)

    # List all outputs
    pdb_id = report['pdb_id']

    print("\n--- Matplotlib Figures ---")
    output_files = sorted(Path(output_dir).glob('Figure*.png'))
    print(f"Generated {len(output_files)} PNG figures:")
    for f in output_files:
        print(f"  - {f.name}")

    print("\n--- PyMOL Scripts ---")
    pml_files = sorted(Path(output_dir).glob('*.pml'))
    print(f"Generated {len(pml_files)} PyMOL scripts:")
    for f in pml_files:
        print(f"  - {f.name}")

    print("\n--- Movie Generation ---")
    print(f"Movie scripts ready in: {output_dir}/{pdb_id}_movies/")
    print(f"Run: bash {pdb_id}_generate_movies.sh")

    print("\n--- Quick Start ---")
    print(f"  1. Master visualization:  pymol {pdb_id}_PRISM4D_master.pml")
    print(f"  2. Pharma analysis:       pymol {pdb_id}_pharma_actionable.pml")
    print(f"  3. Generate movies:       bash {pdb_id}_generate_movies.sh")
    print(f"  4. Publication panels:    pymol {pdb_id}_figure_panels.pml")


if __name__ == '__main__':
    main()
