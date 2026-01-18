# Curated 14 Structure Analysis

## Raw Structure Summary

| PDB  | Atoms | Chains | Residues | Type | Size Tier | Complexity |
|------|-------|--------|----------|------|-----------|------------|
| 1L2Y | 11552 | 1 (A) | 20 | NMR Ensemble (Trp-cage) | Small* | Tier 1 |
| 2F4J | 2328 | 1 (A) | 287 | Single-chain | Small | Tier 1 |
| 3HEC | 2684 | 1 (A) | 329 | Single-chain | Small | Tier 1 |
| 3SQQ | 2340 | 1 (A) | 288 | Single-chain | Small | Tier 1 |
| 4QWO | 2214 | 2 (A,B) | 259 | Homodimer | Small | Tier 1 |
| 6LU7 | 2387 | 2 (A,C) | 309 | SARS-CoV-2 Protease | Small | Tier 2 |
| 1AKE | 3317 | 2 (A,B) | 428 | Adenylate Kinase | Medium | Tier 2 |
| 4Z0J | 3842 | 2 (A,B) | 491 | Heterodimer | Medium | Tier 2 |
| 1HXY | 4739 | 4 (A,B,C,D) | 585 | Tetramer | Medium | Tier 2 |
| 6M0J | 6419 | 2 (A,E) | 791 | SARS-CoV-2 RBD-ACE2 | Large | Tier 3 |
| 2VWD | 6510 | 2 (A,B) | 823 | Nipah Virus | Large | Tier 3 |
| 4J1G | 8093 | 5 (A,B,C,D,E) | 955 | Pentamer | Large | Tier 3 |
| 5IRE | 10940 | 6 (A,B,C,D,E,F) | 1728 | Hexamer (Viral) | XL | Tier 3 |
| 4B7Q | 11976 | 4 (A,B,C,D) | 1548 | Large Complex | XL | Tier 3 |

*Note: 1L2Y is an NMR ensemble with multiple models - actual protein is 20 residues

## Categorization

### By Size Tier
- **Small** (<3000 atoms): 1L2Y*, 2F4J, 3HEC, 3SQQ, 4QWO, 6LU7
- **Medium** (3000-5000 atoms): 1AKE, 4Z0J, 1HXY
- **Large** (5000-10000 atoms): 6M0J, 2VWD, 4J1G
- **XL** (>10000 atoms): 5IRE, 4B7Q

### By Chain Architecture
- **Single-chain**: 1L2Y, 2F4J, 3HEC, 3SQQ
- **Homodimer**: 4QWO, 1AKE
- **Heterodimer**: 4Z0J, 6LU7, 6M0J, 2VWD
- **Multi-chain (4+)**: 1HXY, 4B7Q, 4J1G, 5IRE

### By Structure Family
- **Enzymes**: 1AKE (kinase), 6LU7 (protease)
- **Viral proteins**: 2VWD (Nipah), 5IRE (viral), 6M0J (SARS-CoV-2)
- **Model proteins**: 1L2Y (Trp-cage miniprotein)
- **Multi-chain complexes**: 1HXY, 4B7Q, 4J1G

## Validation Test Selection

### Tier 1 - Simple (2 picks)
1. **3SQQ** - Single chain, 288 residues, clean structure
2. **4QWO** - Homodimer, 259 residues, tests multi-chain routing

### Tier 2 - Medium (2 picks)
1. **6LU7** - SARS-CoV-2 protease, medically relevant, 2 chains
2. **1AKE** - Adenylate kinase, classic benchmark protein

### Tier 3 - Complex (2 picks)
1. **6M0J** - SARS-CoV-2 RBD-ACE2, 791 residues, heterodimer
2. **4J1G** - Pentamer, 5 chains, tests complex routing

**Total: 6 structures for validation benchmark**
