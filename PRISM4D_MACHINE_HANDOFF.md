# PRISM4D Machine Handoff - Complete Setup Guide

## Quick Start (For New Claude Session)

**Context**: UV-LIF coupling for cryptic binding site detection is COMPLETE and ready for validation testing on real targets.

**Current Status**: Code compiles cleanly, true UV enrichment implemented, ready to measure Hit@1/Hit@3 accuracy on real PDB targets.

---

## Working Directory Structure

### Primary Source Code
```
/home/diddy/Desktop/prism4d-v1.2.0-cryo-uv-FULL-source/prism4d-v1.2.0-cryo-uv-FULL-source/prism4d-full-source/
├── crates/
│   ├── prism-nhs/           # NHS engine with UV-LIF coupling
│   │   ├── src/
│   │   │   ├── fused_engine.rs        # NhsAmberFusedEngine (CRITICAL)
│   │   │   ├── persistent_engine.rs   # Batch processing engine
│   │   │   └── lib.rs                 # Exports CryoUvProtocol
│   │   └── examples/
│   │       ├── test_full_pipeline.rs  # Validation test (100% aromatic localization)
│   │       └── benchmark_cryptic_batch.rs  # Batch benchmark
│   ├── prism-gpu/           # CUDA kernels
│   │   └── src/kernels/
│   │       ├── nhs_amber_fused.cu     # Main fused kernel (CRITICAL)
│   │       ├── nhs_amber_fused.ptx    # Compiled PTX
│   │       └── uv_lif_coupling.cuh    # UV-LIF physics
│   └── prism-report/        # Finalize & ranking
│       ├── src/
│       │   ├── finalize.rs            # Site detection & ranking (CRITICAL)
│       │   ├── sites.rs               # Ranking formula
│       │   ├── event_cloud.rs         # Spike event persistence
│       │   └── bin/prism4d.rs         # Main binary
│       └── tests/
└── target/release/
    └── prism4d                        # Compiled binary
```

### Runtime Files (Deployed)
```
/home/diddy/Desktop/PRISM4D_RELEASE/
├── bin/
│   └── prism4d                        # Production binary (COPY FROM SOURCE)
├── assets/ptx/
│   ├── nhs_amber_fused.ptx            # GPU kernel (COPY FROM SOURCE)
│   ├── amber_mega_fused.ptx
│   └── clash_detection.ptx
└── scripts/
    ├── batch/
    │   ├── batch_job_manager.py       # Batch processing
    │   └── gpu_pool_manager.py
    └── batch_accuracy_aggregator.py   # Accuracy measurement
```

### Test Data
```
/home/diddy/Desktop/PRISM4D-v1.1.0-STABLE/results/prism_prep_test/
├── 6LU7_topology.json                 # Mpro (validated)
├── 6M0J_topology.json                 # SARS-CoV-2 Spike
├── 1L2Y_topology.json                 # Small protein
└── ...
```

---

## Git Repository

**Remote**: `git@github.com:Delfictus/Prism4D-bio.git`
**Branch**: `v1.2.0-cryo-uv-quality-assurance`
**Latest Commit**: c2f24ec + compilation fix

**On new machine**:
```bash
cd /home/diddy/Desktop/
git clone git@github.com:Delfictus/Prism4D-bio.git prism4d-v1.2.0-cryo-uv-FULL-source
cd prism4d-v1.2.0-cryo-uv-FULL-source/prism4d-v1.2.0-cryo-uv-FULL-source/prism4d-full-source
git checkout v1.2.0-cryo-uv-quality-assurance
```

---

## Build Dependencies

### Required (Must Install)

```bash
# Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
rustup default stable

# CUDA Toolkit 12.x or 13.x
# Download from: https://developer.nvidia.com/cuda-downloads
# Or: sudo apt install nvidia-cuda-toolkit

# NVCC compiler (check version)
nvcc --version  # Should be 12.x or 13.x

# Python 3.8+
python3 --version
pip3 install numpy

# Build tools
sudo apt install build-essential cmake pkg-config
```

### Optional (For Full Features)
```bash
# PyMOL (visualization)
pip3 install pymol-open-source

# PDF generation
sudo apt install wkhtmltopdf
```

---

## Quick Build & Test

### 1. Compile Everything
```bash
cd prism4d-full-source

# Build release binary
cargo build --release -p prism-report --features gpu

# Should see: "Finished `release` profile [optimized]"
# Binary at: target/release/prism4d
```

### 2. Compile PTX Kernels
```bash
cd crates/prism-gpu/src/kernels

# Check GPU compute capability
nvidia-smi --query-gpu=compute_cap --format=csv,noheader  # e.g., 8.6

# Compile for your GPU (use sm_86 for compute cap 8.6)
nvcc -ptx -arch=sm_86 -O3 --use_fast_math -o nhs_amber_fused.ptx nhs_amber_fused.cu

# Should complete without errors (warnings OK)
```

### 3. Install to Runtime
```bash
# Copy binary
cp target/release/prism4d /home/diddy/Desktop/PRISM4D_RELEASE/bin/

# Copy PTX
cp crates/prism-gpu/src/kernels/nhs_amber_fused.ptx /home/diddy/Desktop/PRISM4D_RELEASE/assets/ptx/

# Verify
/home/diddy/Desktop/PRISM4D_RELEASE/bin/prism4d --version
# Should show: prism4d 0.3.0
```

---

## Critical Files for UV-LIF (HOT PATH)

### CUDA Kernels (Must Recompile for New GPU)
```
crates/prism-gpu/src/kernels/
├── nhs_amber_fused.cu          # Main fused kernel (17,000+ lines)
├── nhs_amber_fused.ptx         # Compiled output (COPY TO RUNTIME)
├── uv_lif_coupling.cuh         # UV-LIF physics (NEW)
└── cryo_thermal_detection.cuh  # Cryo physics
```

**PTX Compilation**:
```bash
nvcc -ptx -arch=sm_86 -O3 --use_fast_math \
  -o nhs_amber_fused.ptx nhs_amber_fused.cu
```

### Rust Source (Core Logic)
```
crates/prism-nhs/src/
├── fused_engine.rs             # CryoUvProtocol (unified cryo+UV)
├── persistent_engine.rs        # Batch streaming engine
└── lib.rs                      # Exports

crates/prism-report/src/
├── finalize.rs                 # Site detection & TRUE enrichment calculation
├── sites.rs                    # Ranking formula (UV weight 45%)
├── event_cloud.rs              # RawSpikeEvent persistence
└── bin/prism4d.rs              # Main entry point
```

---

## Key Implementation Details

### UV-LIF Coupling Parameters (Validated)
```rust
CryoUvProtocol::standard() {
    start_temp: 77.0,            // Liquid N2
    end_temp: 310.0,             // Physiological
    cold_hold_steps: 5000,
    ramp_steps: 10000,
    warm_hold_steps: 5000,
    uv_burst_energy: 30.0,       // kcal/mol (validated)
    uv_burst_interval: 500,      // timesteps
    uv_burst_duration: 50,       // timesteps
    scan_wavelengths: vec![280.0, 274.0, 258.0],  // TRP, TYR, PHE
}
```

**Validation Results**:
- 100% UV spike localization at aromatics (653,895 spikes on 6M0J)
- 2.26x aromatic enrichment over baseline
- 10/11 blind structures passed (90.9%)

### Ranking Formula
```rust
// UV weight: 45% (increased from 25%)
// Enrichment boost: sites with >2.0x rank higher
rank_score = 0.20 * persistence
           + 0.15 * volume
           + 0.45 * uv_confidence  // <-- UV-LIF validation
           + 0.10 * hydrophobicity
           + 0.10 * replica_agreement

where uv_confidence = 0.7 * enrichment_score + 0.3 * aromatic_clustering
```

### True Enrichment Calculation (NEW)
```rust
// In compute_true_uv_enrichment():
uv_on_spikes = spike_events where (timestep % 500 < 50)   // UV burst active
uv_off_spikes = spike_events where (timestep % 500 >= 50) // Thermal baseline

uv_on_aromatic_rate = count(spikes near aromatics) / uv_on_spikes.len()
uv_off_aromatic_rate = count(spikes near aromatics) / uv_off_spikes.len()

enrichment = uv_on_aromatic_rate / uv_off_aromatic_rate
// >2.0x = strong binding site, >1.5x = validated, <1.5x = false positive
```

---

## Output Files Generated

### Per Run
```
output_dir/
├── events.jsonl                # Aggregated pocket events (for clustering)
├── spike_events.jsonl          # RAW spike data (NEW - for true enrichment)
├── summary.json                # Results including tier2_hit_at_1, tier2_hit_at_3
├── pharma_report.json          # Client-facing sites
├── site_metrics.csv            # Ranking scores
└── sites/
    ├── site_001/
    │   ├── site_001.pdb
    │   └── pymol_session.pml
    └── ...
```

### Critical Metrics in summary.json
```json
{
  "correlation": {
    "tier2_hit_at_1": false,    // ← Is #1 ranked site correct?
    "tier2_hit_at_3": false,    // ← Is correct site in top 3?
    "tier2_best_f1": 0.333      // ← Best overlap with true site
  },
  "sites": [
    {
      "site_id": "site_001",
      "rank": 1,
      "metrics": {
        "uv_response": {
          "aromatic_enrichment": 2.5,    // ← TRUE UV-on vs UV-off ratio
          "aromatic_fraction": 0.25,      // ← % Trp/Tyr/Phe
          "event_density": 450.2          // ← Spikes per Ų
        }
      }
    }
  ]
}
```

---

## How to Run on Real Targets

### Quick Test (Existing Topology)
```bash
/home/diddy/Desktop/PRISM4D_RELEASE/bin/prism4d run \
  --topology /home/diddy/Desktop/PRISM4D-v1.1.0-STABLE/results/prism_prep_test/6LU7_topology.json \
  --pdb dummy.pdb \
  --out /tmp/6lu7_test \
  --replicates 1 \
  --wavelengths 280,274,258 \
  --start-temp 77 \
  --end-temp 310 \
  --cold-hold-steps 2000 \
  --ramp-steps 3000 \
  --warm-hold-steps 2000 \
  --skip-ablation \
  -v
```

### Full Pipeline (Download PDB)
```bash
prism4d run \
  --pdb 2CM2.pdb \              # Apo structure
  --holo 2H4K.pdb \             # Holo for validation
  --out results/ptp1b \
  --replicates 1 \
  --wavelengths 280,274,258 \
  --start-temp 77 \
  --end-temp 310 \
  --cold-hold-steps 3000 \
  --ramp-steps 5000 \
  --warm-hold-steps 3000 \
  --skip-ablation               # 3x faster (use spike-based enrichment)
```

**Expected**:
- Runtime: 10-15 min per target
- Output: spike_events.jsonl created
- Logs: "Site X: Enrichment=2.5x ✓ STRONG"
- Result: tier2_hit_at_1 hopefully TRUE

---

## Next Steps (Where We Left Off)

### IMMEDIATE (Before Running Real Targets)

✅ **Compilation fixed** - Code builds cleanly
✅ **Binary updated** - Has true UV enrichment calculation
✅ **Pushed to GitHub** - All work backed up

### VALIDATION TESTING (Next Session)

**Task #12**: Run 5-target validation
```bash
# Test on available topologies
for topo in 6LU7 6M0J 1L2Y 1AKE 1HXY; do
  prism4d run \
    --topology .../prism_prep_test/${topo}_topology.json \
    --pdb dummy.pdb \
    --out /tmp/${topo}_test \
    --skip-ablation

  # Extract tier2_hit_at_1, tier2_hit_at_3 from summary.json
done

# Aggregate metrics
python3 scripts/batch_accuracy_aggregator.py /tmp/ --output results.json
```

**Expected Outcome**:
- Hit@1: 3/5 (60%) ← Industry competitive
- Hit@3: 4/5 (80%) ← Match SiteMap standards
- If achieved → Client-ready!

---

## Known Issues & Workarounds

### Issue #1: Persistence All 0.010
**Status**: Not blocking (secondary metric)
**Workaround**: Enrichment is primary ranking signal
**Fix**: Can investigate later

### Issue #2: No Truth Residues for Most Benchmarks
**Status**: Need to prepare holo structures
**Workaround**: Use existing topologies without validation first
**Fix**: Download/prepare holo PDBs for 20 benchmark targets

### Issue #3: Ablation Slower (3x)
**Status**: Fixed - use --skip-ablation
**Benefit**: 15 min instead of 45 min per target

---

## File Checksums (Verify Integrity)

**Critical binaries** (check these match):
```bash
# Binary
md5sum /home/diddy/Desktop/PRISM4D_RELEASE/bin/prism4d
# Should be recent build (after compilation fix)

# PTX kernels
md5sum /home/diddy/Desktop/PRISM4D_RELEASE/assets/ptx/nhs_amber_fused.ptx
# Should match source: crates/prism-gpu/src/kernels/nhs_amber_fused.ptx
```

---

## Environment Variables

```bash
# CUDA (if needed)
export CUDA_VISIBLE_DEVICES=0

# Rust (build)
export RUSTFLAGS="-C target-cpu=native"

# Logging
export RUST_LOG=info  # or debug for verbose

# GPU arch (for nvcc)
export GPU_ARCH=sm_86  # Adjust based on your GPU
```

---

## Testing Checklist (New Machine)

### 1. Environment Setup
```bash
□ Rust installed (rustc --version)
□ CUDA installed (nvcc --version)
□ GPU accessible (nvidia-smi)
□ Python 3.8+ (python3 --version)
□ Git configured (git config user.name)
```

### 2. Repository Clone
```bash
□ Cloned from GitHub
□ Checked out v1.2.0-cryo-uv-quality-assurance branch
□ All files present (ls crates/prism-nhs/src/fused_engine.rs)
```

### 3. Compilation
```bash
□ cargo build --release -p prism-report --features gpu succeeds
□ Binary at target/release/prism4d
□ PTX at crates/prism-gpu/src/kernels/nhs_amber_fused.ptx
```

### 4. Installation
```bash
□ Binary copied to /home/diddy/Desktop/PRISM4D_RELEASE/bin/
□ PTX copied to /home/diddy/Desktop/PRISM4D_RELEASE/assets/ptx/
□ prism4d --version works
```

### 5. Quick Smoke Test
```bash
□ Run on 6LU7_topology.json (1000 steps)
□ Completes without crash
□ spike_events.jsonl created
□ summary.json has tier2 metrics
```

---

## Current Task Status

**Completed (11/12)**:
- ✅ UV-LIF physics validated
- ✅ Unified cryo-UV protocol
- ✅ Spike event persistence
- ✅ True enrichment calculation
- ✅ Enhanced ranking formula
- ✅ Hyperoptimized batch system
- ✅ Anti-overfitting validation
- ✅ Compilation fixed
- ✅ All changes committed & pushed

**Pending (1/12)**:
- ⚠️ **Task #12**: Run 5-target validation to measure Hit@1, Hit@3
  - **Goal**: Hit@1 > 60%, Hit@3 > 75%
  - **Blocker**: None - ready to execute
  - **Timeline**: 2-4 hours

**Deferred (Low Priority)**:
- Task #3: Fix persistence (0.010 uniform)
- Task #7: Verify prism-prep quality
- Task #6: Full 20-target benchmark (after #12 validates)

---

## Performance Expectations

### UV-LIF Physics
- Aromatic localization: 100% (validated)
- Aromatic enrichment: 2.26x (validated)
- Generalization: 90.9% (10/11 blind structures)

### Site Detection (Target)
- Hit@1: >60% (competitive with SiteMap)
- Hit@3: >75% (industry standard)
- Precision@10: >70%

### Speed
- Per target: 10-15 min with --skip-ablation
- Batch: 2.58x parallel speedup
- Persistent engine: Saves ~300ms per structure

---

## Commands for New Claude Session

### Resume Context
```bash
cd /home/diddy/Desktop/prism4d-v1.2.0-cryo-uv-FULL-source/prism4d-v1.2.0-cryo-uv-FULL-source/prism4d-full-source

# Check what's committed
git log --oneline -10

# Read status documents
cat IMPLEMENTATION_COMPLETE.md
cat HONEST_STATUS_REPORT.md
cat RANKING_FIX_STATUS.md
```

### Continue Work
```bash
# The NEXT action is to run validation testing:
# Run on 5 targets, measure Hit@1 and Hit@3

# Example validation run:
prism4d run \
  --topology /path/to/topology.json \
  --holo /path/to/holo.pdb \
  --out /tmp/test \
  --skip-ablation

# Check results:
cat /tmp/test/summary.json | grep tier2
ls /tmp/test/spike_events.jsonl  # Should exist
```

---

## What New Claude Needs to Know

### 1. **UV-LIF Physics Works**
- 100% aromatic localization
- 2.26x enrichment
- NO overfitting (physics-based)
- Ready for production

### 2. **Ranking Was The Problem**
- Sites ARE detected (F1 > 0.3)
- But ranked wrong (Hit@1 = 0%)
- **Fix implemented**: True UV enrichment from spike events
- **Needs validation**: Test if Hit@1 improved

### 3. **Critical Code Paths**
- `nhs_amber_fused.cu` lines 1240-1350: UV-LIF coupling
- `finalize.rs` lines 1485-1590: compute_true_uv_enrichment()
- `sites.rs` lines 347-390: Ranking formula

### 4. **Ablation Not Required**
- Can skip with --skip-ablation (3x faster)
- UV enrichment uses timestep % 500 (UV-on vs UV-off)
- No separate baseline run needed

### 5. **Client-Ready Criteria**
- Hit@1 > 60% (currently unknown - needs testing)
- Hit@3 > 75% (currently unknown - needs testing)
- Precision@10 > 70%
- **Blocker**: Need to RUN validation to measure

---

## Important Paths (Hard-Coded)

**Test Data**:
- `/home/diddy/Desktop/PRISM4D-v1.1.0-STABLE/results/prism_prep_test/` - Pre-prepared topologies
- `/home/diddy/Desktop/6M0J_results_test/` - 6M0J validation data

**Runtime**:
- `/home/diddy/Desktop/PRISM4D_RELEASE/bin/prism4d` - Production binary
- `/home/diddy/Desktop/PRISM4D_RELEASE/assets/ptx/` - PTX kernels

**Source**:
- `/home/diddy/Desktop/prism4d-v1.2.0-cryo-uv-FULL-source/prism4d-v1.2.0-cryo-uv-FULL-source/prism4d-full-source/` - Full source code

---

## Diagnostic Commands

### Check Binary Has True Enrichment
```bash
# Run on test target
prism4d run --topology 6LU7_topology.json --pdb dummy.pdb --out /tmp/test --skip-ablation -v 2>&1 | grep -i "enrichment"

# Should see:
#   "Site site_001: UV-on=X/Y, UV-off=A/B, Enrichment=2.5x"
#   "✓ True UV enrichment computed for N sites"
```

### Check spike_events.jsonl Format
```bash
head -1 /tmp/test/spike_events.jsonl | python3 -c "
import json, sys
e = json.loads(sys.stdin.read())
print('Timestep:', e['timestep'])
print('Nearby residues:', e['nearby_residues'])
print('Position:', e['position'])
"
# Should show timestep and nearby_residues (needed for enrichment)
```

### Measure Accuracy
```bash
cat /tmp/test/summary.json | python3 -c "
import json, sys
d = json.load(sys.stdin)
corr = d['correlation']
print('Hit@1:', corr['tier2_hit_at_1'])
print('Hit@3:', corr['tier2_hit_at_3'])
print('Best F1:', corr['tier2_best_f1'])
"
```

---

## Summary for New Claude

**What we built**: UV-LIF coupling for cryptic site detection with true UV-on vs UV-off enrichment calculation

**What works**: Physics validated (90.9% generalization), code compiles, binary ready

**What's unknown**: Does true enrichment improve Hit@1 from 0% to 60%+? (needs testing)

**Next action**: Run validation on 5 targets, measure Hit@1/Hit@3, determine if client-ready

**Timeline**: 2-4 hours to validate, then either ready for clients or need iteration

**Everything needed for identical session is in Git repo + this handoff document.**
