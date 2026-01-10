# Chemistry-Aware GNM (CA-GNM) Implementation Plan

## The Core Problem

Standard GNM uses uniform spring constants:
```
Γᵢⱼ = -1 if dist(i,j) < cutoff
Γᵢⱼ = 0  otherwise
```

This treats ALL contacts equally - a salt bridge the same as a van der Waals contact,
a Proline the same as a Glycine. **Chemistry is ignored.**

## The Solution: Chemistry-Weighted Kirchhoff Matrix

Instead of binary contacts, use **chemistry-aware spring constants**:

```
Γᵢⱼ = -k(i,j) × w_dist(dᵢⱼ) × w_burial(i,j) × w_hbond(i,j) × w_type(i,j)
```

Where:
- `k(i,j)` = base spring constant from amino acid pair
- `w_dist` = distance weighting (already implemented)
- `w_burial` = burial depth weighting (buried contacts are stiffer)
- `w_hbond` = hydrogen bond detection (H-bonds are stiffer)
- `w_type` = contact type (backbone-backbone vs sidechain-sidechain)

---

## Component 1: Amino Acid Pair Stiffness Matrix

Different amino acid pairs have different intrinsic stiffness when in contact.

```rust
/// 20x20 symmetric matrix of pairwise stiffness factors
/// Based on statistical potentials and MD force field parameters
pub const AA_STIFFNESS: [[f32; 20]; 20] = [
    // ALA, ARG, ASN, ASP, CYS, GLN, GLU, GLY, HIS, ILE, ...
    // Diagonal: self-interactions
    // Off-diagonal: pair interactions
];

/// Key patterns:
/// - GLY-GLY: 0.6 (both flexible → weak spring)
/// - PRO-PRO: 1.4 (both rigid → stiff spring)
/// - CYS-CYS: 2.0 (potential disulfide → very stiff)
/// - Hydrophobic pairs: 1.2 (stable core contacts)
/// - Charged opposite: 1.3 (salt bridge potential)
/// - Charged same: 0.7 (repulsion → weak effective spring)
```

### Residue Intrinsic Flexibility (from literature)

| Residue | Flexibility Factor | Reason |
|---------|-------------------|--------|
| GLY | 1.40 | No sidechain, maximum backbone freedom |
| ALA | 0.90 | Small sidechain, relatively rigid |
| PRO | 0.55 | Ring constrains backbone φ angle |
| SER | 1.05 | Small polar, moderate flexibility |
| THR | 0.95 | Branched β-carbon constrains |
| CYS | 0.85 | Can form disulfides |
| VAL | 0.85 | Branched β-carbon |
| ILE | 0.85 | Branched, hydrophobic core |
| LEU | 0.90 | Hydrophobic, moderate |
| MET | 1.00 | Long flexible sidechain |
| PHE | 0.80 | Rigid aromatic ring |
| TYR | 0.80 | Rigid aromatic + hydroxyl |
| TRP | 0.75 | Largest aromatic, very rigid |
| HIS | 0.90 | Aromatic, can be charged |
| LYS | 1.15 | Long flexible charged sidechain |
| ARG | 1.10 | Long charged sidechain |
| ASP | 1.00 | Short charged |
| GLU | 1.05 | Longer charged |
| ASN | 1.05 | Short polar |
| GLN | 1.10 | Longer polar |

**Pair stiffness**: `k(i,j) = 1.0 / (flex[i] × flex[j])`
- GLY-GLY: 1.0 / (1.4 × 1.4) = 0.51 (weak spring)
- PRO-PRO: 1.0 / (0.55 × 0.55) = 3.3 (very stiff)
- TRP-PRO: 1.0 / (0.75 × 0.55) = 2.4 (stiff)

---

## Component 2: Burial Depth Weighting

Contacts between buried residues are more stable (lower fluctuation).

```rust
/// Compute burial depth from Cα neighbor counting
fn compute_burial(ca_positions: &[[f32; 3]], radius: f64) -> Vec<f64> {
    // Count neighbors within radius (typically 10-12Å)
    // Normalize by max possible neighbors
    // Returns 0.0 (surface) to 1.0 (core)
}

/// Burial weighting for spring constant
/// Buried contacts = stiffer springs
fn burial_weight(burial_i: f64, burial_j: f64) -> f64 {
    let avg_burial = (burial_i + burial_j) / 2.0;
    // Buried contacts are 20-40% stiffer
    1.0 + 0.4 * avg_burial  // Range: 1.0 to 1.4
}
```

---

## Component 3: Hydrogen Bond Detection

H-bonds provide additional stability beyond van der Waals contacts.

```rust
/// Detect backbone hydrogen bonds from Cα geometry
/// N-H···O=C pattern: Cα(i) to Cα(j) with |i-j| ≥ 3
fn detect_hbonds(ca_positions: &[[f32; 3]]) -> Vec<(usize, usize)> {
    let mut hbonds = Vec::new();

    for i in 0..n {
        for j in (i + 3)..n {  // At least 3 residues apart
            let dist = distance(ca_positions[i], ca_positions[j]);

            // Helix: ~5.4Å for i,i+4
            // Sheet: ~4.5-5.5Å for antiparallel, ~6.5Å for parallel

            if dist > 4.0 && dist < 7.0 {
                // Check geometry consistent with H-bond
                // (simplified - full detection needs N,H,O,C atoms)

                // Helix pattern: i,i+3 or i,i+4 at specific distances
                if (j == i + 3 && dist > 4.5 && dist < 5.5) ||
                   (j == i + 4 && dist > 5.0 && dist < 6.5) {
                    hbonds.push((i, j));
                }

                // Sheet pattern: long-range at ~4.5-5.5Å
                if j > i + 4 && dist > 4.0 && dist < 6.0 {
                    // Additional check: local backbone geometry
                    hbonds.push((i, j));
                }
            }
        }
    }
    hbonds
}

/// H-bond contacts get stiffer springs
fn hbond_weight(is_hbond: bool) -> f64 {
    if is_hbond { 1.5 } else { 1.0 }  // 50% stiffer for H-bonds
}
```

---

## Component 4: Contact Type Classification

Not all contacts are equal - backbone-backbone vs sidechain interactions.

```rust
enum ContactType {
    BackboneBackbone,   // Cα-Cα through backbone (sequential or H-bond)
    SidechainSidechain, // Cβ-Cβ type interactions
    BackboneSidechain,  // Mixed
    SequenceLocal,      // |i-j| ≤ 4 (along chain)
    LongRange,          // |i-j| > 12 (tertiary contacts)
}

fn classify_contact(i: usize, j: usize, dist: f64) -> ContactType {
    let seq_sep = (j as i32 - i as i32).abs() as usize;

    if seq_sep <= 4 {
        ContactType::SequenceLocal
    } else if seq_sep > 12 {
        ContactType::LongRange
    } else if dist < 6.0 {
        ContactType::BackboneBackbone
    } else {
        ContactType::SidechainSidechain
    }
}

fn contact_type_weight(contact_type: ContactType) -> f64 {
    match contact_type {
        ContactType::SequenceLocal => 1.2,    // Chain connectivity is stiff
        ContactType::BackboneBackbone => 1.1, // Backbone contacts stable
        ContactType::LongRange => 1.3,        // Tertiary contacts important
        ContactType::SidechainSidechain => 0.9, // More dynamic
        ContactType::BackboneSidechain => 1.0,
    }
}
```

---

## Component 5: Salt Bridge Detection

Oppositely charged residues in proximity form stabilizing salt bridges.

```rust
const POSITIVE: &[&str] = &["ARG", "LYS", "HIS"];
const NEGATIVE: &[&str] = &["ASP", "GLU"];

fn is_salt_bridge(res_i: &str, res_j: &str, dist: f64) -> bool {
    if dist > 8.0 { return false; }  // Salt bridges typically < 8Å

    let i_pos = POSITIVE.contains(&res_i);
    let i_neg = NEGATIVE.contains(&res_i);
    let j_pos = POSITIVE.contains(&res_j);
    let j_neg = NEGATIVE.contains(&res_j);

    (i_pos && j_neg) || (i_neg && j_pos)
}

fn salt_bridge_weight(is_salt_bridge: bool) -> f64 {
    if is_salt_bridge { 1.4 } else { 1.0 }  // 40% stiffer
}
```

---

## The Complete CA-GNM Algorithm

```rust
pub fn build_chemistry_aware_kirchhoff(
    ca_positions: &[[f32; 3]],
    residue_names: &[&str],
    cutoff: f64,
    sigma: f64,
) -> DMatrix<f64> {
    let n = ca_positions.len();
    let cutoff_sq = cutoff * cutoff;
    let sigma_sq = sigma * sigma;

    // Pre-compute burial depths
    let burial = compute_burial(ca_positions, 10.0);

    // Detect hydrogen bonds
    let hbonds: HashSet<(usize, usize)> = detect_hbonds(ca_positions)
        .into_iter().collect();

    let mut kirchhoff = DMatrix::zeros(n, n);

    for i in 0..n {
        for j in (i + 1)..n {
            let dist_sq = distance_squared(ca_positions[i], ca_positions[j]);

            if dist_sq < cutoff_sq {
                let dist = dist_sq.sqrt();

                // 1. Distance weighting (Gaussian decay)
                let w_dist = (-dist_sq / (2.0 * sigma_sq)).exp();

                // 2. Amino acid pair stiffness
                let flex_i = residue_flexibility(residue_names[i]);
                let flex_j = residue_flexibility(residue_names[j]);
                let w_aa = 1.0 / (flex_i * flex_j);

                // 3. Burial weighting
                let w_burial = burial_weight(burial[i], burial[j]);

                // 4. Hydrogen bond detection
                let is_hbond = hbonds.contains(&(i, j)) || hbonds.contains(&(j, i));
                let w_hbond = hbond_weight(is_hbond);

                // 5. Salt bridge detection
                let is_salt = is_salt_bridge(residue_names[i], residue_names[j], dist);
                let w_salt = salt_bridge_weight(is_salt);

                // 6. Contact type
                let contact_type = classify_contact(i, j, dist);
                let w_type = contact_type_weight(contact_type);

                // Combined spring constant
                let k = w_dist * w_aa * w_burial * w_hbond * w_salt * w_type;

                kirchhoff[(i, j)] = -k;
                kirchhoff[(j, i)] = -k;
            }
        }
    }

    // Set diagonal (sum of row)
    for i in 0..n {
        let row_sum: f64 = kirchhoff.row(i).iter().sum();
        kirchhoff[(i, i)] = -row_sum;
    }

    kirchhoff
}
```

---

## Expected Improvements

| Component | Expected Δρ | Rationale |
|-----------|-------------|-----------|
| AA pair stiffness | +0.02-0.04 | GLY/PRO properly handled |
| Burial weighting | +0.01-0.02 | Core vs surface differentiated |
| H-bond detection | +0.01-0.02 | Secondary structure stabilization |
| Salt bridges | +0.005-0.01 | Electrostatic stabilization |
| Contact types | +0.01-0.02 | Tertiary contacts weighted |
| **Combined** | **+0.05-0.10** | **ρ = 0.67-0.72** |

---

## Implementation Order

1. **Phase A: AA Pair Stiffness** (~200 lines)
   - Implement residue flexibility table
   - Modify Kirchhoff matrix construction
   - Test: expect +0.02-0.04

2. **Phase B: Burial Weighting** (~100 lines)
   - Neighbor counting for burial depth
   - Weight spring constants by burial
   - Test: expect additional +0.01-0.02

3. **Phase C: H-bond Detection** (~150 lines)
   - Geometric H-bond detection from Cα
   - Stiffer springs for H-bonded pairs
   - Test: expect additional +0.01-0.02

4. **Phase D: Salt Bridges** (~50 lines)
   - Charge-based detection
   - Additional stiffness for salt bridges
   - Test: expect additional +0.005-0.01

5. **Phase E: Integration & Tuning** (~100 lines)
   - Combine all components
   - Tune weight factors on training subset
   - Validate on held-out proteins

---

## File Structure

```
crates/prism-physics/src/
├── gnm_chemistry.rs      # NEW: Chemistry-aware GNM
├── residue_chemistry.rs  # NEW: AA properties, flexibility factors
├── contact_analysis.rs   # NEW: H-bond, salt bridge detection
├── burial_analysis.rs    # NEW: Burial depth computation
└── lib.rs               # Add module exports
```

---

## Key Design Principles

1. **Modify the Kirchhoff matrix, not the output**
   - Previous approach: RMSF × factor = broken
   - New approach: Γᵢⱼ × factor = physics-correct

2. **All factors are multiplicative on spring constants**
   - Stiffer spring → lower RMSF (correctly propagated through eigenvalues)
   - No post-hoc corrections that break physics

3. **Chemistry is encoded in spring constants, not corrections**
   - GLY-GLY contact: weak spring → high RMSF
   - PRO-PRO contact: stiff spring → low RMSF
   - This is how real MD force fields work!

4. **Still pure physics, no ML**
   - All parameters from literature/force fields
   - Interpretable: can explain why each residue is flexible
   - Fast: same eigenvalue decomposition, just better matrix

---

## Validation Plan

1. **Ablation**: Test each component individually
2. **Sensitivity**: Check robustness to parameter choices
3. **Generalization**: 5-fold cross-validation on AlphaFlow-82
4. **Target**: Mean ρ ≥ 0.70, or statistically significant improvement over baseline
