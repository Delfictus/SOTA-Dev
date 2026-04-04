# PRISM-AI: Cryptic Binding Site Predictor

Predict cryptic (hidden) protein binding sites from a single PDB file. No molecular dynamics simulation required.

## Quick Start

```bash
# Install dependencies
pip install torch esm prody scipy scikit-learn mdtraj biopython

# Run prediction
python predict.py your_protein.pdb --output results.json --visualize pockets.pml
```

## What It Does

PRISM-AI predicts which residues in a protein are part of cryptic binding pockets - sites that are hidden in the apo (unbound) structure but open upon ligand binding or conformational change.

**Input:** Any `.pdb` file (apo structure)
**Output:** Per-residue binding probability + ranked binding site predictions
**Time:** ~10 seconds on GPU, ~60 seconds on CPU

## How It Works

The model is a 109-member ensemble distilled from the PRISM-4D neuromorphic molecular dynamics engine. During training, the engine simulated 3.6 billion spike events across 174 proteins to learn which structural features predict pocket opening. The student model learned to approximate these physics signals from structure alone.

### Feature Pipeline (automatic, no user action needed)

| Feature | Dims | Source | Compute Time |
|---------|------|--------|-------------|
| Amino acid identity | 20 | PDB | instant |
| Hydrophobicity | 1 | Kyte-Doolittle | instant |
| Secondary structure | 3 | DSSP via mdtraj | <1s |
| Solvent accessibility | 1 | SASA via mdtraj | <1s |
| B-factor | 1 | PDB ATOM records | instant |
| Normal mode analysis | 26 | ProDy ANM (20 modes) | ~2s |
| ESM-2 embeddings | 1280 | esm2_t33_650M_UR50D | ~5s (GPU) |
| **Total** | **1332** | | **~10s** |

## Usage

### Basic

```bash
python predict.py 1abc.pdb
```

### With options

```bash
# Specific chain
python predict.py 1abc.pdb --chain A

# Save results
python predict.py 1abc.pdb --output results.json

# Generate PyMOL visualization
python predict.py 1abc.pdb --visualize pockets.pml

# Top 3 sites only
python predict.py 1abc.pdb --top-k 3

# All options
python predict.py 1abc.pdb --chain A --top-k 5 --output results.json --visualize viz.pml
```

### From Python

```python
from predict import predict

result = predict("1abc.pdb", chain="A", top_k=5)

# Per-residue probabilities
for resid, prob in result["per_residue_probabilities"].items():
    if prob > 0.3:
        print(f"Residue {resid}: {prob:.3f}")

# Ranked sites
for site in result["top_sites"]:
    print(f"Site {site['rank']}: {site['n_residues']} residues, "
          f"probability={site['mean_prob']:.3f}")
```

## PDB Preparation

The predictor handles raw PDB files directly. For best results:

1. **Use the biological assembly**, not the asymmetric unit
2. **Remove non-protein chains** if you only care about one chain (use `--chain`)
3. **Apo structures preferred** — the model predicts cryptic pockets that are HIDDEN in the unbound form
4. **Resolution < 3.0 Angstrom** recommended
5. **No preprocessing needed** — the script handles altconfs, missing residues, non-standard amino acids

### If your PDB has issues

```bash
# Download fresh from RCSB
curl -s "https://files.rcsb.org/download/1ABC.pdb" -o 1abc.pdb

# Run on specific chain
python predict.py 1abc.pdb --chain A
```

## Output Format

```json
{
  "input_pdb": "1abc.pdb",
  "chain": "A",
  "n_residues": 263,
  "n_ensemble_models": 109,
  "n_sites_found": 7,
  "top_sites": [
    {
      "rank": 1,
      "residue_ids": [45, 47, 48, 51, 53, 67, 69, 71, 89, 91, 94, 96],
      "centroid": [12.4, 138.2, 9.7],
      "n_residues": 12,
      "mean_prob": 0.342,
      "max_prob": 0.458,
      "sum_prob": 4.104,
      "spatial_extent": 14.2,
      "rank_score": 0.4521
    }
  ],
  "per_residue_probabilities": {
    "1": 0.1823,
    "2": 0.1956,
    "3": 0.3421,
    ...
  },
  "model_version": "prism-ai-student-v002"
}
```

## Requirements

- Python 3.9+
- PyTorch 2.0+
- [ESM-2](https://github.com/facebookresearch/esm) (downloaded automatically on first run, ~2.5GB)
- ProDy
- scipy, scikit-learn
- mdtraj (optional, for DSSP/SASA — falls back to defaults if missing)
- BioPython (optional)

### Install all at once

```bash
pip install torch esm prody scipy scikit-learn mdtraj biopython
```

### GPU

CUDA GPU strongly recommended. ESM-2 embedding computation is ~10x faster on GPU.

## Model Details

| Property | Value |
|----------|-------|
| Architecture | MLP ensemble (512→512→256→1) |
| Ensemble size | 109 LOTO folds |
| Input features | 1332 (52 structural + 1280 ESM-2) |
| Training data | 174 proteins, 3.6B spike events |
| Training AUROC | 0.639 (LOTO cross-validation) |
| Blind detection | 47.7% residue overlap @ top-5 (107 targets) |
| Inference time | ~10s/protein (GPU) |

## File Structure

```
prism-ai-inference/
├── predict.py          # Main inference script
├── models/             # 109 fold model weights (~600MB)
│   ├── student_fold_000_1a4u.pt
│   ├── student_fold_001_1a8d.pt
│   └── ... (109 files)
├── examples/           # Example PDB files + expected output
└── README.md           # This file
```

## Citation

PRISM-4D neuromorphic engine and PRISM-AI distilled predictor.
Delfictus Inc. 2026.

## License

Proprietary. Contact info@delfictus.com for licensing.
