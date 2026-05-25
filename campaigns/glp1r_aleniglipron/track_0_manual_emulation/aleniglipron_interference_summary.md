## Aleniglipron Thermodynamic Field Interference Analysis

### Layer 1: Whole-Molecule Scoring
- Total Pi_clash across 9 critical edges: 41.8023
- Total Pi_complement: 46.1713
- Projected durability score: 546.6557 +/- 60.9397
- Confidence class: moderate

### Layer 2: Fragment Decomposition (BRICS)
- Fragments identified: 8
- Dominant liability fragment: FRAG-C ([13*]c1c2c(nn1[19*])CCN([31*])[C@@H]2C, 20.2% of total clash)
- Inter-fragment coupling: 2.4970

### Per-Edge Attribution
| Edge | Whole Clash | Sum Fragment Clash | Coupling | Dominant Fragment | Dominant Fraction |
|---|---:|---:|---:|---|---:|
| glp1r_5VEX_WT:PHE143->ILE147 | 21.1308 | 19.7931 | 1.3376 | FRAG-C | 20.2% |
| glp1r_5VEX_WT:PHE143->TYR148 | 0.0000 | 0.0000 | 0.0000 | FRAG-A | 0.0% |
| glp1r_5VEX_WT:TYR148->TYR152 | 0.0000 | 0.0000 | 0.0000 | FRAG-A | 0.0% |
| glp1r_5VEX_WT:TYR152->TYR148 | 20.6716 | 19.5122 | 1.1593 | FRAG-C | 20.6% |
| glp1r_6LN2_A316T:TYR242->ILE313 | 0.0000 | 0.0000 | 0.0000 | FRAG-A | 0.0% |
| glp1r_6XOX_WT:ARG421->TRP417 | 0.0000 | 0.0000 | 0.0000 | FRAG-A | 0.0% |
| glp1r_6XOX_WT:PHE169->SER416 | 0.0000 | 0.0000 | 0.0000 | FRAG-A | 0.0% |
| glp1r_6XOX_WT:TRP417->ARG421 | 0.0000 | 0.0000 | 0.0000 | FRAG-A | 0.0% |
| glp1r_6XOX_WT:TYR241->ARG310 | 0.0000 | 0.0000 | 0.0000 | FRAG-A | 0.0% |


### Structural Rationales
**glp1r_5VEX_WT:PHE143->ILE147:** At glp1r_5VEX_WT:PHE143->ILE147, aleniglipron shows Pi_clash = 21.13. FRAG-C is the dominant contributor (4.26, 20% of total). Inter-fragment coupling contributes 1.34 additional clash, indicating synergistic steric compression when fragments are connected.

**glp1r_5VEX_WT:PHE143->TYR148:** Aleniglipron shows negligible steric interference at glp1r_5VEX_WT:PHE143->TYR148 (Pi_clash = 0.000). No fragment contributes significant clash.

**glp1r_5VEX_WT:TYR148->TYR152:** Aleniglipron shows negligible steric interference at glp1r_5VEX_WT:TYR148->TYR152 (Pi_clash = 0.000). No fragment contributes significant clash.

**glp1r_5VEX_WT:TYR152->TYR148:** At glp1r_5VEX_WT:TYR152->TYR148, aleniglipron shows Pi_clash = 20.67. FRAG-C is the dominant contributor (4.26, 21% of total). Inter-fragment coupling contributes 1.16 additional clash, indicating synergistic steric compression when fragments are connected.

**glp1r_6LN2_A316T:TYR242->ILE313:** Aleniglipron shows negligible steric interference at glp1r_6LN2_A316T:TYR242->ILE313 (Pi_clash = 0.000). No fragment contributes significant clash.

**glp1r_6XOX_WT:ARG421->TRP417:** Aleniglipron shows negligible steric interference at glp1r_6XOX_WT:ARG421->TRP417 (Pi_clash = 0.000). No fragment contributes significant clash.

**glp1r_6XOX_WT:PHE169->SER416:** Aleniglipron shows negligible steric interference at glp1r_6XOX_WT:PHE169->SER416 (Pi_clash = 0.000). No fragment contributes significant clash.

**glp1r_6XOX_WT:TRP417->ARG421:** Aleniglipron shows negligible steric interference at glp1r_6XOX_WT:TRP417->ARG421 (Pi_clash = 0.000). No fragment contributes significant clash.

**glp1r_6XOX_WT:TYR241->ARG310:** Aleniglipron shows negligible steric interference at glp1r_6XOX_WT:TYR241->ARG310 (Pi_clash = 0.000). No fragment contributes significant clash.

### Recommendation
Prioritize modifications to FRAG-C if reducing receptor-side steric clash is the objective; protect fragments with low dominant clash unless potency SAR requires changes.