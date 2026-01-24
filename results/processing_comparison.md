# Per-Chain vs Whole-Structure Processing Comparison

## Overview

Compared two processing approaches for 5 large multi-chain viral glycoproteins:
1. **Whole-Structure**: Process entire multi-chain structure through Stage 1 & 2 together
2. **Per-Chain**: Split into individual chains, process separately, combine topologies

## Energy Comparison

| Structure | Atoms | Whole-Structure | Per-Chain | Change | Result |
|-----------|-------|-----------------|-----------|--------|--------|
| 1HXY      | 9,444 | 3.11e12         | 3.93e11   | 7.9x ↓ | **IMPROVED** |
| 2VWD      | 12,926| 1.11e12         | 3.13e14   | 282x ↑ | **WORSE** |
| 4B7Q      | 23,312| 1.42e10         | 5.53e9    | 2.6x ↓ | **IMPROVED** |
| 5IRE      | 26,297| 4.43e10         | 3.06e10   | 1.4x ↓ | **IMPROVED** |
| 6M0J      | 12,510| 7.32e8          | 3.92e8    | 1.9x ↓ | **IMPROVED** |

## Summary

- **4/5 structures improved** with per-chain processing
- **1/5 structures degraded** dramatically (2VWD)

## Analysis

### Structures That Improved (4/5)

| Structure | Improvement | Notes |
|-----------|-------------|-------|
| 1HXY | 7.9x | Rhinovirus VP (3 chains) - Most improved |
| 4B7Q | 2.6x | RSV F glycoprotein (6 chains) |
| 5IRE | 1.4x | Zika NS1 (6 chains) |
| 6M0J | 1.9x | SARS-CoV-2 Spike RBD (2 chains) |

Per-chain processing helps by:
- Avoiding inter-chain clashes during OpenMM minimization
- Each chain minimized in isolation reaches better local minimum
- Combining pre-minimized chains preserves individual chain quality

### Structure That Degraded (2VWD)

| Structure | Degradation | Notes |
|-----------|-------------|-------|
| 2VWD | 282x worse | Nipah G glycoprotein (4 chains) |

**Root Cause Analysis for 2VWD:**
- 2VWD is Nipah virus attachment glycoprotein (HeV/NiV)
- Chains A-D form a homo-tetramer with CRITICAL inter-chain contacts
- When processed separately:
  - Each chain lacks stabilizing contacts from neighbors
  - Interface residues minimize to unnatural conformations
  - When recombined, these create severe steric clashes
- This structure REQUIRES whole-structure processing to maintain quaternary stability

## Recommendations

### Processing Strategy by Structure Type

| Structure Type | Recommended Approach | Reason |
|----------------|---------------------|--------|
| Heteromeric complexes | Per-chain | Chains have independent roles |
| Homo-oligomers (tight) | Whole-structure | Quaternary contacts essential |
| RBD + antibody | Per-chain | Independent binding partners |
| Spike trimers | Whole-structure | Trimeric interface critical |

### Hybrid Strategy

For production, implement a hybrid approach:
1. Analyze structure quaternary state (BIOMT records, symmetry)
2. If homo-oligomer with tight interface → whole-structure
3. If heteromeric or loose complex → per-chain
4. If energy still high after preferred method → try alternative

## Energy Thresholds

Based on successful structures, reasonable energy ranges:

| Size Category | Atoms | Normal Range | Warning |
|---------------|-------|--------------|---------|
| Small | <5,000 | <1e7 | >1e8 |
| Medium | 5-15,000 | <1e9 | >1e10 |
| Large | 15-30,000 | <1e10 | >1e11 |
| XL | >30,000 | <5e10 | >1e11 |

## Final Status

After processing optimization:

| Structure | Final Energy | Status | Method Used |
|-----------|--------------|--------|-------------|
| 1HXY | 3.93e11 | ⚠️ Elevated | Per-chain |
| 2VWD | 1.11e12 | ⚠️ Elevated | Whole-structure (best) |
| 4B7Q | 5.53e9 | ✅ Normal | Per-chain |
| 5IRE | 3.06e10 | ⚠️ Elevated | Per-chain |
| 6M0J | 3.92e8 | ✅ Normal | Per-chain |

**Production Ready: 2/5** (4B7Q, 6M0J)
**Needs Further Work: 3/5** (1HXY, 2VWD, 5IRE)

## Next Steps for Elevated Structures

1. **1HXY (Rhinovirus)**: Try extended minimization cycles
2. **2VWD (Nipah)**: Requires custom restraints on inter-chain contacts
3. **5IRE (Zika NS1)**: May benefit from per-dimer processing (3 dimers)
