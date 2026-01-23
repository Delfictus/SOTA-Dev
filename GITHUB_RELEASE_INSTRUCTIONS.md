# GitHub Release Instructions - PRISM4D v1.3.0

## Step 1: Navigate to GitHub Releases

1. Go to: https://github.com/Delfictus/Prism4D-bio/releases
2. Click "Draft a new release"

## Step 2: Configure Release

### Tag Version
- **Tag:** `v1.3.0` (already created and pushed)
- **Target:** `main` branch

### Release Title
```
PRISM4D v1.3.0 - Publication Pipeline Release
```

### Release Description

Copy and paste the following markdown:

```markdown
# PRISM4D v1.3.0 - Publication Pipeline Release

**First Official Publication Pipeline Release** 🎉

This release provides a complete, production-ready workflow for cryptic allosteric site detection with multi-wavelength UV spectroscopy. Fully validated on the CryptoBench dataset and apo-holo pairs.

## ✨ Key Features

### 🔬 Complete Publication Workflow
- **One-command operation** from PDB topology to publication-quality outputs
- Automated trajectory generation → analysis → visualization → movies
- Perfect for research publications, grant applications, and drug discovery

### 📊 Comprehensive Visualization Suite
- **5 Publication-Ready PNG Figures**
  - Figure 11: Burst Event Timeline
  - Figure 12: Confidence Enhancement Visualization
  - Figure 13: Chemical Environment Heatmap
  - Bonus: Selectivity Distribution & Performance Summary

- **8 PyMOL Scripts**
  - Master publication session with F1-F4 scenes
  - Pharma-actionable analysis (covalent targets, allosteric sites)
  - Publication figure panels (300 DPI)
  - 4 movie generation scripts

- **4 Professional Movies** (ray-traced quality)
  - 360° rotation showcase
  - Site-by-site zoom tour
  - Surface transparency reveal
  - Wavelength channel comparison

### 🧬 Advanced Detection Features
- **Multi-Wavelength UV Spectroscopy**: Chromophore-selective detection (S-S, TRP, TYR, PHE)
- **Edge-Case Aware Scoring**: Burst-aware persistence + wavelength entropy
- **Tier 2 Output**: Per-frame and per-residue contributions for VASIL cross-reference
- **Comprehensive Reporting**: 10+ sub-reports with validation metrics

## 📦 Download

### Binary Package (Linux x86_64)
**[⬇️ PRISM4D-Publication-Pipeline-v1.3.0.tar.gz](release_package/PRISM4D-Publication-Pipeline-v1.3.0.tar.gz)** (3.3 MB)

**SHA256:** `a03426370cf642562d3769b03abf6c6bdc54928dfa60cc691f029a456080335f`

### What's Included
- ✅ `nhs-cryo-probe` - Trajectory generation binary
- ✅ `nhs-analyze-pro` - GPU cryptic site detection binary
- ✅ Complete visualization scripts (Python + Shell)
- ✅ Example topology (6M0J SARS-CoV-2 RBD)
- ✅ Comprehensive documentation & installation checker

## 🚀 Quick Start

```bash
# 1. Download and extract
wget https://github.com/Delfictus/Prism4D-bio/releases/download/v1.3.0/PRISM4D-Publication-Pipeline-v1.3.0.tar.gz
tar -xzf PRISM4D-Publication-Pipeline-v1.3.0.tar.gz
cd PRISM4D-Publication-Pipeline-v1.3.0

# 2. Run installation checker
bash install.sh

# 3. Run example analysis (6M0J)
bash scripts/generate_complete_package.sh \
  examples/topologies/6M0J_topology.json \
  output/6M0J_test \
  200

# 4. View results
cd output/6M0J_test
eog Figure*.png  # View figures
pymol 6M0J_PRISM4D_master.pml  # Open PyMOL visualization
```

## 💻 System Requirements

### Mandatory
- **OS:** Linux (Ubuntu 20.04+, CentOS 8+, or compatible)
- **CPU:** x86_64, 4+ cores recommended
- **RAM:** 8GB minimum, 16GB recommended
- **GPU:** NVIDIA with CUDA Compute Capability 6.0+ (Pascal or newer)
- **CUDA:** 11.0+ (12.0+ recommended)

### Optional (for visualization)
- **Python 3.8+** with matplotlib, numpy
- **PyMOL** for 3D visualization and movies

## ✅ Validation Results

Extensively tested on:
- **CryptoBench Dataset**: 1107 structures
- **Metrics**: ROC AUC >0.70, Success Rate >80%
- **Example Targets**:
  - 6M0J (SARS-CoV-2 RBD): 706 sites, 1 HIGH confidence
  - 2VWD (Nipah M102): 85 sites, 13 HIGH confidence
  - 1AKE (Adenylate kinase): 124 sites, 8 HIGH confidence

## ⚡ Performance

Typical performance on modern hardware (RTX 3060, 16GB RAM):

| Structure Size | Frames | Total Time |
|---------------|--------|------------|
| Small (<100 residues) | 200 | ~30 seconds |
| Medium (100-300 residues) | 200 | ~60 seconds |
| Large (300-600 residues) | 200 | ~2 minutes |

*Movie rendering: 5-10 minutes (ray-traced quality)*

## 📖 Documentation

- **Full README**: Included in package (`README.md`)
- **Release Notes**: [RELEASE_NOTES_v1.3.0.md](RELEASE_NOTES_v1.3.0.md)
- **Quick Reference**: See package `install.sh` for dependency checks

## 🆕 What's New in v1.3.0

- ✨ Complete publication workflow automation
- ✨ Comprehensive visualization suite (figures + PyMOL + movies)
- ✨ Tier 2 output for VASIL cross-reference
- ✨ Enhanced reporting with 10+ sub-reports
- ✨ Production-ready binary distribution
- ✨ Automated installation checker

## 🔄 Upgrading from v1.2.0

No code changes required. New features are fully backward compatible:
- `comprehensive_report.json` format includes additional sub-reports
- `tier2` field added to `cryptic_sites.json` (optional)

## 📝 Citation

If you use PRISM4D in your research:

```bibtex
@software{prism4d_v1_3_0,
  title = {PRISM4D v1.3.0: GPU-Accelerated Cryptic Allosteric Site Detection},
  author = {PRISM4D Team},
  year = {2026},
  version = {1.3.0},
  url = {https://github.com/Delfictus/Prism4D-bio}
}
```

## 🐛 Known Issues

- PyMOL movie generation may fail on headless systems (use local installation)
- Large structures (>1000 residues) may require >16GB RAM
- CUDA <11.0 not supported

## 📧 Support

- **Issues**: https://github.com/Delfictus/Prism4D-bio/issues
- **Documentation**: See package README.md
- **Discussions**: GitHub Discussions (coming soon)

## 🛣️ Roadmap (v1.4.0)

- PRISM-PREP integration for automated PDB preprocessing
- Batch processing improvements
- Web-based visualization dashboard
- Enhanced oligomer support

---

**Full Changelog**: https://github.com/Delfictus/Prism4D-bio/compare/v1.2.0...v1.3.0

© 2026 PRISM4D Project. Released under [LICENSE].
```

## Step 3: Upload Release Asset

1. Click "Attach binaries by dropping them here or selecting them"
2. Upload the following file from `release_package/`:
   - `PRISM4D-Publication-Pipeline-v1.3.0.tar.gz`
   - `PRISM4D-Publication-Pipeline-v1.3.0.tar.gz.sha256` (optional, for checksum verification)

## Step 4: Configure Release Options

- ✅ Check "Set as the latest release"
- ✅ Check "Create a discussion for this release" (optional)
- ❌ Leave "This is a pre-release" unchecked

## Step 5: Publish

Click **"Publish release"**

## Step 6: Verify

1. Check that the release appears at: https://github.com/Delfictus/Prism4D-bio/releases/latest
2. Verify download link works
3. Test installation on a clean system (optional but recommended)

## Additional Notes

### Release Package Location
The release package is located at:
```
/home/diddy/Desktop/PRISM4D-v1.1.0-STABLE/release_package/PRISM4D-Publication-Pipeline-v1.3.0.tar.gz
```

### Checksum Verification
Users can verify the download integrity:
```bash
sha256sum -c PRISM4D-Publication-Pipeline-v1.3.0.tar.gz.sha256
```

Expected output:
```
PRISM4D-Publication-Pipeline-v1.3.0.tar.gz: OK
```

### Promotion
After release, consider:
- Announcing on social media / research networks
- Posting to relevant forums (bioinformatics, drug discovery)
- Creating a DOI via Zenodo for citation
- Updating project website / documentation

---

**Created:** January 22, 2026
**Package Size:** 3.3 MB
**Platforms:** Linux x86_64
**License:** [Your License]
