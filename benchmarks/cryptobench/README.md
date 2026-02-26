# CryptoBench Benchmark for PRISM4D

**Reference:** Skrhak et al., *Bioinformatics* 2025, 41(1):btae745
**Dataset:** 1,107 apo structures with 5,493 cryptic binding pockets (pRMSD >= 2.0A)
**Test set:** 222 structures (10% sequence identity clustering, no train/test leakage)

## Quick Start

```bash
# All commands from this directory:
cd benchmarks/cryptobench

# Step 1: Download CryptoBench dataset (~1.2 GB total)
bash 01_download_cryptobench.sh

# Step 2: Prepare topologies (CIF→PDB + prism-prep)
# Full dataset (~1,107 structures, ~2-3 hours):
conda run -n prism_dock python3 02_prepare_topologies.py

# Or test set only (222 structures, ~30-60 min):
conda run -n prism_dock python3 02_prepare_topologies.py --test-only

# Step 3: Generate batched run commands (groups of 5)
python3 03_generate_run_commands.py

# Step 4: Run the benchmark (DO THIS OUTSIDE CLAUDE CODE)
bash run_all_batches.sh          # All batches
bash batches/batch_001.sh        # Or one batch at a time

# Step 5: Evaluate results
python3 04_evaluate_results.py
```

## Pipeline Structure

```
01_download_cryptobench.sh    — Downloads from OSF (dataset.json + folds.json + CIF files)
02_prepare_topologies.py      — CIF→PDB, prism-prep topology, ground truth extraction
03_generate_run_commands.py   — Generates batch_NNN.sh scripts (5 per batch)
04_evaluate_results.py        — DCA/DCC/residue overlap evaluation vs ground truth
```

## Run Command

Each structure runs with:
```
RUST_LOG=info nhs_rt_full -t <topo>.json -o <results_dir> \
    --fast --hysteresis --multi-stream 8 \
    --spike-percentile 95 --rt-clustering -v
```

## Estimated Runtime

| Subset | Structures | Est. Time (RTX 5080) |
|--------|-----------|---------------------|
| Test set | 222 | ~15-30 hours |
| Full dataset | 1,107 | ~75-150 hours |

(Based on ~4-26 min per structure from existing benchmarks)

## Evaluation Metrics

The evaluation script (04) computes:
- **DCA** (Distance to Closest Atom) at < 4A threshold
- **DCC** (Distance Center-to-Center) at < 4A, 8A, 10A thresholds
- **Top-1 / Top-3 / Top-N+2** success rates
- **Per-residue** precision, recall, F1
- **Runtime** statistics

## Published Baselines (from CryptoBench paper)

| Method | AUC | AUPRC | MCC |
|--------|-----|-------|-----|
| pLM-NN (sequence) | 0.86 | 0.36 | 0.39 |
| P2Rank (structure, apo) | 0.81 | 0.21 | 0.27 |
| PocketMiner (structure) | 0.76 | 0.19 | 0.22 |

## Directory Layout (after running)

```
benchmarks/cryptobench/
  data/
    dataset.json          # CryptoBench ground truth (8.4 MB)
    folds.json            # Train/test splits
    cif_files/            # mmCIF structures (1.15 GB)
  structures/             # Converted PDB files
  topologies/             # prism-prep AMBER ff14SB topologies
  ground_truth/           # Per-structure ground truth JSON
  results/                # nhs_rt_full output per structure
  batches/                # Batch shell scripts
  run_manifest.txt        # List of ready structures
  run_all_batches.sh      # Master runner script
  cryptobench_evaluation.json  # Final evaluation results
```
