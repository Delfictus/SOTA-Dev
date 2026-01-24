# PRISM4D Release Registry

This document tracks all stable, tagged releases of PRISM4D modules.

## Release Philosophy

- **Immutable**: Once tagged, a release NEVER changes
- **Semantic Versioning**: MAJOR.MINOR.PATCH
  - MAJOR: Breaking API changes
  - MINOR: New features, backward compatible
  - PATCH: Bug fixes only
- **Recovery**: `git checkout <tag>` restores exact state

---

## Active Releases

### Cryo-UV Pipeline (Primary Research Tools)

| Module | Version | Tag | Date | Status | Key Metrics |
|--------|---------|-----|------|--------|-------------|
| nhs-adaptive | 1.0.0 | `nhs-adaptive-v1.0.0` | 2026-01-22 | STABLE | 200K spikes, 231 sites, 2001 steps/sec |
| nhs-batch | 1.0.0 | `nhs-batch-v1.0.0` | 2026-01-22 | STABLE | 4-structure parallel, 315K atoms/sec |
| nhs-cryo-probe | 1.0.0 | `nhs-cryo-probe-v1.0.0` | 2026-01-22 | STABLE | Cryo @ 100K, UV pump-probe |
| nhs-detect | 1.0.0 | `nhs-detect-v1.0.0` | 2026-01-22 | PARTIAL | Static detection only |

### Ensemble Generation

| Module | Version | Tag | Date | Status | Key Metrics |
|--------|---------|-----|------|--------|-------------|
| generate-ensemble-simd | 1.0.0 | `ensemble-simd-v1.0.0` | 2026-01-22 | STABLE | SIMD batched, 315K atoms/sec |
| generate-ensemble | 1.0.0 | `ensemble-v1.0.0` | 2026-01-22 | STABLE | Standard ensemble generation |

### Preprocessing

| Module | Version | Tag | Date | Status | Key Metrics |
|--------|---------|-----|------|--------|-------------|
| prism-prep | 1.2.0 | `prism-prep-v1.2.0` | 2026-01-22 | PRODUCTION | AMBER reduce, ff14SB topology |

### Benchmarking

| Module | Version | Tag | Date | Status | Key Metrics |
|--------|---------|-----|------|--------|-------------|
| cryptic-benchmark-v2 | 2.0.0 | `cryptobench-v2.0.0` | 2026-01-22 | STABLE | ROC 0.445, 84.8% success |
| atlas-benchmark | 1.0.0 | `atlas-v1.0.0` | 2026-01-22 | STABLE | 96.3% pass, ρ=0.850 |

### Core Libraries

| Crate | Version | Tag | Date | Status |
|-------|---------|-----|------|--------|
| prism-gpu | 1.1.0 | `prism-gpu-v1.1.0` | 2026-01-22 | STABLE |
| prism-nhs | 1.0.0 | `prism-nhs-v1.0.0` | 2026-01-22 | STABLE |
| prism-physics | 1.0.0 | `prism-physics-v1.0.0` | 2026-01-22 | STABLE |
| prism-io | 1.0.0 | `prism-io-v1.0.0` | 2026-01-22 | STABLE |
| prism-core | 1.0.0 | `prism-core-v1.0.0` | 2026-01-22 | STABLE |

---

## How to Use

### Checkout a Specific Release
```bash
# Restore nhs-adaptive to exact v1.0.0 state
git checkout nhs-adaptive-v1.0.0

# Create a branch to work on enhancements
git checkout -b enhance-nhs-adaptive nhs-adaptive-v1.0.0
```

### Download Pre-built Binary (GitHub Releases)
```bash
# Download from GitHub releases
gh release download nhs-adaptive-v1.0.0 --pattern "nhs-adaptive"

# Or via URL
wget https://github.com/USER/PRISM4D/releases/download/nhs-adaptive-v1.0.0/nhs-adaptive
```

### Compare Versions
```bash
# See what changed between versions
git diff nhs-adaptive-v1.0.0..nhs-adaptive-v1.1.0

# View release notes
gh release view nhs-adaptive-v1.0.0
```

---

## Version History

### nhs-adaptive

#### v1.0.0 (2026-01-22)
- Initial stable release
- Cryo-UV pump-probe protocol
- LIF spike detection (threshold 0.5)
- Signal tuning: 10x amplification, 1.5x tau
- Results: 200K spikes, 231 correlated cryptic sites

### prism-prep

#### v1.2.0 (2026-01-22)
- Self-contained preprocessing binary
- AMBER reduce integration
- Cryptic/escape mode routing
- Strict validation mode

#### v1.1.0 (2026-01-15)
- Added batch processing
- Glycan handling improvements

#### v1.0.0 (2026-01-10)
- Initial release
- Basic PDB sanitization

---

## Creating a New Release

### 1. Update Version Numbers
```bash
# Edit Cargo.toml
vim crates/prism-nhs/Cargo.toml
# Change: version = "1.0.0" → version = "1.1.0"
```

### 2. Update Changelog
```bash
# Add entry to RELEASES.md under the module
```

### 3. Commit and Tag
```bash
git add -A
git commit -m "release: nhs-adaptive v1.1.0 - [description]"
git tag -a nhs-adaptive-v1.1.0 -m "Description of changes"
git push origin main --tags
```

### 4. Create GitHub Release
```bash
gh release create nhs-adaptive-v1.1.0 \
  --title "NHS Adaptive v1.1.0" \
  --notes "## Changes\n- Feature 1\n- Feature 2" \
  ./target/release/nhs-adaptive
```

---

## Safety Guarantees

1. **Tags are immutable** - Git prevents modifying tagged commits
2. **Releases are permanent** - GitHub releases cannot be silently edited
3. **Binaries are archived** - Pre-compiled executables preserved
4. **Source snapshots** - Full source at tag point recoverable

Even if `main` branch is completely rewritten, tagged releases remain accessible forever.
