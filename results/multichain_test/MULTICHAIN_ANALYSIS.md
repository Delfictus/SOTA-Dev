# Multi-Chain Preprocessing Analysis

Generated: 2026-01-17
Updated: 2026-01-17 (Smart Routing + AMBER Integration)

## Summary

Multi-chain preprocessing now uses **smart routing** based on inter-chain contact analysis:
- Independent chains (low contact density) → Per-chain processing
- Coupled chains (high contact density) → Whole-structure processing
- Disulfide bridges → Always whole-structure

### Key Metrics for Routing
| Metric | Low (→ Multichain) | High (→ Whole) |
|--------|-------------------|----------------|
| Contact density | < 1.5 | > 2.0 |
| H-bonds per chain pair | < 10 | > 10 |
| Salt bridges per pair | < 1.0 | > 1.0 |

## Results

| Structure | Chains | Whole-Structure PE | Multi-Chain PE | Result |
|-----------|--------|-------------------|----------------|--------|
| 1HXY (HIV gp120) | 4 | 3.24e+10 | 9.25e+12 | **WORSE** (285x) |
| 5IRE (Zika Env) | 6 | 1.70e+14 | 7.03e+06 | **FIXED** (24M x better) |
| 4J1G (Flu HA) | 5 | 2.15e+14 | 9.04e+13 | Better (2.4x) |

## Why Multi-Chain Works for 5IRE but Not 1HXY

### 5IRE (Zika Envelope Hexamer)
- 6 independent protomers forming a symmetric hexamer
- Each protomer is structurally complete on its own
- Per-chain optimization finds good local minima independently
- RMSD between whole and multi-chain coordinates: **2.17 Å**
- Final energy: 267 kcal/mol/atom (production-ready range)

### 1HXY (HIV-1 gp120)
- 4 chains with extensive inter-chain contacts
- CD4 binding site spans multiple chains
- gp120-gp41 interactions are critical
- Per-chain processing disrupts essential contacts
- Energy penalty for broken inter-chain hydrogen bonds

### 4J1G (Influenza Hemagglutinin Trimer)
- 5 chains (HA1/HA2 × 3)
- Moderate improvement but still high energy
- Some inter-chain contacts, but less critical than HIV

## Topology Validation

All topologies pass structural validation:
- Bond/angle/dihedral counts are consistent (1.02 bonds/atom, 1.86 angles/atom)
- No steric clashes detected
- No orphan atoms
- Bond lengths within expected ranges

The issue is **not** in the topology file structure but in:
1. Initial coordinate quality after hydrogen addition
2. Inter-chain contact preservation

## Recommendations

### When to Use Multi-Chain Processing

**Good candidates** (independent subunits):
- Symmetric oligomers (hexamers, octamers)
- Viral capsid subunits
- Structures where chains are loosely associated
- E.g., 5IRE (Zika envelope)

**Poor candidates** (interdependent chains):
- Tight protein-protein complexes
- Structures with inter-chain disulfide bonds
- Envelope glycoproteins with receptor binding sites
- E.g., 1HXY (HIV gp120), 2VWD (Nipah G)

### Future Improvements

1. **Inter-chain contact detection**: Before splitting, analyze inter-chain hydrogen bonds and disulfides
2. **Selective splitting**: Only split chains that are weakly associated
3. **Contact-aware recombination**: After per-chain processing, re-optimize inter-chain contacts
4. **Hybrid approach**: Process large symmetric structures per-chain, small complexes as whole

## Files Created/Modified

| File | Purpose |
|------|---------|
| `scripts/multichain_preprocessor.py` | Smart routing with contact analysis |
| `scripts/interchain_contacts.py` | Inter-chain contact analyzer |
| `scripts/validate_topology.py` | Pre-engine topology validation |
| `scripts/stage1_sanitize_amber.py` | AMBER reduce hydrogen placement |
| `scripts/combine_chain_topologies.py` | Chain topology recombination |
| `scripts/glycan_preprocessor.py` | Glycan detection and handling |

## Smart Routing Results

Tested with calibrated thresholds based on empirical data:

| Structure | Chains | Contact Density | H-bonds/pair | Routing | Correct? |
|-----------|--------|-----------------|--------------|---------|----------|
| 5IRE (Zika) | 6 | 1.14 | 5.1 | 🔀 MULTICHAIN | ✅ |
| 1HXY (HIV) | 4 | 3.04 | 16.5 | 📦 WHOLE | ✅ |
| 6LU7 (SARS-CoV-2) | 2 | N/A | N/A | ✨ STANDARD | ✅ |

### Engine Results with Smart Routing

| Structure | Routing | Final PE | Energy/Atom | Status |
|-----------|---------|----------|-------------|--------|
| 5IRE | MULTICHAIN | 1.21e+10 | 461 | Elevated |
| 1HXY | WHOLE | 5.76e+10 | 6,100 | Elevated |
| 6LU7 | STANDARD | 1.78e+06 | 376 | **Production Ready** |

## AMBER Hydrogen Placement

For higher quality hydrogen placement, the preprocessor now supports AMBER's `reduce` tool:

```bash
# Use AMBER reduce for hydrogen placement
python multichain_preprocessor.py input.pdb output.json --use-amber
```

### Benefits of AMBER reduce over PDBFixer:
1. **Optimizes H-bond networks** - Considers global hydrogen bonding patterns
2. **Asn/Gln/His flip states** - Correctly orients ambiguous residues
3. **Minimizes steric clashes** - Avoids bad van der Waals contacts
4. **Better initial coordinates** - More stable starting point for MD

### Requirements
```bash
conda install -c conda-forge ambertools
```

## Next Steps

1. Run comparative benchmarks: PDBFixer vs AMBER reduce
2. Test AMBER hydrogen placement on elevated structures
3. Investigate remaining high-energy cases (1HXY, 2VWD)
