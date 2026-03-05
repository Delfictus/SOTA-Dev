# PRISM-4D Manuscript Figures

All figures generated at 300 DPI in PDF format for publication quality.

## Figure Descriptions

### Figure 1: Pipeline Architecture (fig1_pipeline.pdf)
4-stage flowchart showing the complete PRISM-4D pipeline:
- **Stage 1**: NHS Engine (AMBER MD, Cryo-Thermal, UV Excitation)
- **Stage 2**: Spike Detection (3-Channel LIF, UV/LIF/EFP)
- **Stage 3**: SNDC Clustering (RT-DBSCAN, Watershed, Eikonal BFS, Peak Tracking)
- **Stage 4**: Site Scoring (Druggability, Classification, Covalent ID)

**Color scheme**: Teal/blue gradient with clear input (PDB) and output (Druggable Sites) labels.

### Figure 2: Cryo-Thermal Hysteresis Protocol (fig2_hysteresis.pdf)
Temperature vs simulation progress showing the 5-phase cryo-thermal cycle:
- Phase 1: Cold hold at 50K (0-25.5%)
- Phase 2: Ramp up 50→300K (25.5-36.4%)
- Phase 3: Warm hold at 300K (36.4-63.6%)
- Phase 4: Ramp down 300→50K (63.6-74.5%)
- Phase 5: Cold return at 50K (74.5-100%)

**Features**: UV burst markers (teal triangles every ~0.45%), physiological reference line at 300K, orange heating/purple cooling phases.

### Figure 3: DCC Benchmark Bar Chart (fig3_benchmark.pdf)
12-protein benchmark results showing Distance to Closest Crystal site (DCC):
- **Excellent (<5Å)**: BACE1 (3.6Å), TEM1 (3.7Å), KRAS (3.8Å), PTP1B (4.8Å)
- **Good (<8Å)**: AdSS (6.0Å), Abl (6.2Å), IL-2 (6.3Å), SIRPα (7.1Å)
- **Marginal (<10Å)**: ERα (9.5Å), HIV-1 (9.8Å), FKBP12 (9.8Å)

**Color coding**: Green (excellent), teal (good), orange (marginal) with threshold reference lines.

### Figure 4: Detection Accuracy Summary (fig4_accuracy.pdf)
Stacked accuracy metrics:
- **<5Å**: 4/11 (36.4%)
- **<8Å**: 8/11 (72.7%)
- **<10Å**: 11/11 (100%)

**Clean bar chart** showing progressive improvement across distance thresholds.

### Figure 5: Reproducibility Analysis (fig5_reproducibility.pdf)
Three-panel figure demonstrating PRISM-4D's reproducibility:
- **Panel A**: Centroid position scatter (5 seeds) showing ±0.06Å variation
- **Panel B**: Spike count variation (CV = 0.2%, ~3241±5 spikes)
- **Panel C**: Pairwise DCC histogram (max = 0.06Å vs C-C bond 1.40Å reference)

**Demonstrates**: Near-deterministic behavior across random seeds.

### Figure 6: Method Comparison Capability Matrix (fig6_capability.pdf)
Capability heatmap comparing PRISM-4D against 5 existing methods (FTMap, fpocket, P2Rank, DeepSite, PocketMiner):

**Capabilities assessed**:
- Physics Simulation
- Cryptic Pocket Detection
- Covalent Identification
- UV Spectroscopy
- Spike Detection
- No Training Data Required

**Legend**: Filled circle (full support), half-filled (partial), empty (none). PRISM-4D shows full support across all capabilities.

### Figure 7: Hardware Cost Comparison (fig7_cost.pdf)
Log-scale horizontal bar chart comparing costs:
- **PRISM-4D (RTX 5080)**: $999
- **Cloud HPC (per run)**: $500
- **Schrödinger (annual license)**: $30,000
- **D.E. Shaw Anton**: $100,000,000

**Demonstrates**: >100,000× cost advantage over specialized hardware.

### Figure 8: SIRPα Head-to-Head (fig8_sirpa.pdf)
Two-panel direct comparison between PRISM-4D and P2Rank on SIRPα (2WNG):
- **Panel A**: DCC comparison (PRISM-4D: 7.1Å vs P2Rank: 9.0Å)
- **Panel B**: Quality metrics (Quality Score, Druggability, Confidence)
  - PRISM-4D: [0.652, 0.652, 1.0]
  - P2Rank: [N/A, 0.77, 0.002]

**Shows**: Superior accuracy and confidence despite lower druggability score.

## Technical Details

### Generation
All figures generated using `/home/diddy/Desktop/Prism4D-bio/prism4d_manuscript/generate_figures.py`

### Settings
- **Resolution**: 300 DPI
- **Format**: PDF with TrueType fonts (Type 42)
- **Font**: Arial/DejaVu Sans, 10pt base
- **Export**: `bbox_inches='tight'` for clean edges

### Color Palette
- Primary: `#2E8B99` (teal)
- Success: `#2ECC71` (green)
- Warning: `#FF8C42` (orange)
- Error: `#E74C3C` (red)
- Neutral: `#95A5A6` (gray)
- Accent: `#9B59B6` (purple)

### Files
```
figures/
├── fig1_pipeline.pdf         (27 KB)
├── fig2_hysteresis.pdf       (23 KB)
├── fig3_benchmark.pdf        (23 KB)
├── fig4_accuracy.pdf         (19 KB)
├── fig5_reproducibility.pdf  (25 KB)
├── fig6_capability.pdf       (22 KB)
├── fig7_cost.pdf             (19 KB)
└── fig8_sirpa.pdf            (27 KB)
```

## Regeneration
To regenerate all figures:
```bash
cd /home/diddy/Desktop/Prism4D-bio/prism4d_manuscript
python3 generate_figures.py
```

## Citation
When using these figures, cite the PRISM-4D manuscript (in preparation).
