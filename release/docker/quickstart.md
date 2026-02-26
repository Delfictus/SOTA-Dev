# PRISM4D Demo — Quick Start Guide

## 1. Run Detection on a Sample Structure

Pre-prepared topologies are in `~/samples/`. Run detection:

```bash
prism4d detect ~/samples/2fhz.topology.json ~/results/2fhz
```

This will:
- Initialize the GPU engine (RTX 5080, Blackwell architecture)
- Run 8-stream cryo-UV molecular dynamics with neuromorphic spike detection
- Cluster spikes using OptiX RT-core acceleration
- Score and rank binding sites by druggability
- Output binding_sites.json with detected cryptic pockets

Typical runtime: 3-8 minutes per structure.

## 2. View Results

```bash
cat ~/results/2fhz/2fhz.binding_sites.json | jq '.sites[0]'
```

Key fields in each site:
- `centroid`: 3D coordinates of the pocket center
- `estimated_volume`: pocket volume in cubic Angstroms
- `quality_score`: physics-informed quality (0-1)
- `druggability`: multi-component druggability assessment
- `lining_residues`: residues forming the pocket wall
- `classification`: Cryptic / Allosteric / Catalytic / Surface

## 3. Batch Processing

```bash
prism4d batch ~/samples/ ~/results/
```

## 4. Re-rank Sites (Post-Processing)

```bash
prism4d rerank ~/results/
```

## 5. Full Pipeline (from PDB)

If you have a PDB file:

```bash
# Step 1: Prepare topology (requires conda env)
prism4d prep my_protein.pdb my_protein.topology.json

# Step 2: Detect pockets
prism4d detect my_protein.topology.json ~/results/my_protein
```

## Architecture Highlights

- **Neuromorphic Holographic Stream (NHS)**: LIF neuron network detects
  pocket opening events via dewetting spike cascades
- **Cryo-UV Protocol**: Aromatic UV excitation (280nm) perturbs TRP/TYR/PHE
  to reveal cryptic conformations in ~8 minutes
- **OptiX RT Clustering**: NVIDIA ray-tracing cores accelerate 3D spatial
  clustering of spike events (O(N) vs O(N^2) DBSCAN)
- **87 CUDA Kernels**: AMBER ff14SB physics fully on GPU, zero CPU round-trips

## Questions?

Contact the PRISM4D team for licensing and integration discussions.
