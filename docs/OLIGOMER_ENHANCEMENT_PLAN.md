# PRISM-Delta Oligomer Enhancement Plan

## Overview

Enhance PRISM-Delta to detect cryptic sites at oligomeric interfaces, validated against known antibody binding sites like m102.4 on Nipah G protein.

## Phase 1: Biological Assembly Support

### 1.1 Parse REMARK 350 Records

PDB files contain REMARK 350 records that define the biological assembly:

```
REMARK 350 BIOMOLECULE: 1
REMARK 350 APPLY THE FOLLOWING TO CHAINS: A, B, C, D
REMARK 350   BIOMT1   1  1.000000  0.000000  0.000000        0.00000
REMARK 350   BIOMT2   1  0.000000  1.000000  0.000000        0.00000
REMARK 350   BIOMT3   1  0.000000  0.000000  1.000000        0.00000
```

**Implementation:**
```rust
pub struct BiologicalAssembly {
    pub biomolecule_id: u32,
    pub chains: Vec<char>,
    pub transforms: Vec<Matrix4x4>,
}

pub fn parse_remark_350(pdb_content: &str) -> Vec<BiologicalAssembly>
```

### 1.2 Preserve Chain Information

Update `PdbAtom` to include chain_id:
```rust
pub struct PdbAtom {
    pub chain_id: char,        // NEW: Chain identifier
    pub residue_id: i32,
    pub residue_name: String,
    // ... existing fields
}
```

### 1.3 Multi-Chain Topology Builder

```rust
pub struct OligomerTopology {
    pub chains: HashMap<char, Vec<PdbAtom>>,
    pub inter_chain_contacts: Vec<(AtomRef, AtomRef)>,
    pub interface_residues: HashMap<char, Vec<i32>>,
}

impl OligomerTopology {
    pub fn from_biological_assembly(pdb: &str, assembly_id: u32) -> Self;
    pub fn detect_interfaces(&self, cutoff: f32) -> Vec<InterfaceRegion>;
}
```

## Phase 2: Inter-Chain Interface Detection

### 2.1 Interface Residue Identification

Residues at chain interfaces are often cryptic sites (buried in monomer, exposed in assembly):

```rust
pub struct InterfaceRegion {
    pub chain_a: char,
    pub chain_b: char,
    pub residues_a: Vec<i32>,
    pub residues_b: Vec<i32>,
    pub buried_sasa: f64,      // SASA lost upon complex formation
    pub interface_area: f64,   // Total interface area (Å²)
}

pub fn detect_interface_residues(
    oligomer: &OligomerTopology,
    sasa_cutoff: f64,  // ≥10 Å² buried
) -> Vec<InterfaceRegion>
```

### 2.2 Interface Cryptic Score Boost

Boost cryptic scores for interface residues:

```rust
pub fn compute_interface_boost(
    residue: i32,
    chain: char,
    interfaces: &[InterfaceRegion],
) -> f64 {
    // Residues that become accessible upon oligomer dissociation
    // are high-value cryptic targets
    if is_at_interface(residue, chain, interfaces) {
        return 0.15;  // +15% boost to cryptic score
    }
    0.0
}
```

## Phase 3: Known Functional Site Proximity Weighting

### 3.1 Functional Site Database

```rust
pub struct FunctionalSite {
    pub name: String,
    pub site_type: SiteType,  // Epitope, ActiveSite, AllostericSite, BindingPocket
    pub residues: Vec<i32>,
    pub chain: char,
    pub source: String,       // Literature reference
}

pub enum SiteType {
    Epitope { antibody: String, affinity_nm: Option<f64> },
    ActiveSite,
    AllostericSite,
    BindingPocket { ligand: String },
}
```

### 3.2 Proximity Scoring

```rust
pub fn compute_functional_proximity(
    residue: i32,
    chain: char,
    coords: &[f32; 3],
    functional_sites: &[FunctionalSite],
) -> f64 {
    let mut max_proximity = 0.0;

    for site in functional_sites {
        let min_dist = compute_min_distance(coords, site);

        // Gaussian decay: sites within 10Å get boost
        let proximity = (-min_dist.powi(2) / (2.0 * 10.0_f64.powi(2))).exp();
        max_proximity = max_proximity.max(proximity * site_weight(site));
    }

    max_proximity
}
```

## Phase 4: Antibody Binding Site Validation

### 4.1 m102.4 Antibody Dataset

The m102.4 antibody binds Nipah G protein at the central cavity:

```rust
pub const M102_4_EPITOPE: &[i32] = &[
    // Core epitope residues (from PDB: 3D11)
    507, 508, 509, 510, 511,  // Central cavity
    529, 530, 531, 532,
    // Extended epitope
    252, 253, 254,
];

pub struct AntibodyValidation {
    pub antibody_name: String,
    pub pdb_complex: String,   // e.g., "3D11" for m102.4-G complex
    pub epitope_residues: Vec<i32>,
    pub kd_nm: Option<f64>,    // Binding affinity
}
```

### 4.2 Validation Metrics

```rust
pub struct AntibodyOverlapMetrics {
    pub epitope_recall: f64,     // % of epitope residues in top predictions
    pub epitope_precision: f64,  // % of top predictions that are epitope
    pub rank_of_first_epitope: usize,  // Rank of first epitope residue
    pub mean_epitope_rank: f64,
}

pub fn validate_against_antibody(
    predictions: &[ResiduePrediction],
    antibody: &AntibodyValidation,
) -> AntibodyOverlapMetrics
```

## Phase 5: Henipavirus Benchmark Dataset

### 5.1 Structure Collection

| PDB ID | Protein | State | Resolution | Notes |
|--------|---------|-------|------------|-------|
| 2VWD | NiV G | Apo | 3.5Å | Nipah attachment glycoprotein |
| 3D11 | NiV G + m102.4 | Holo | 3.0Å | Antibody complex |
| 6VY4 | HeV G | Apo | 2.3Å | Hendra attachment glycoprotein |
| 8XQ3 | NiV G tetramer | Oligomer | 3.2Å | Full biological assembly |
| 6CMG | NiV F | Pre-fusion | 3.2Å | Fusion glycoprotein |

### 5.2 Ground Truth from Literature

```rust
pub fn load_henipavirus_ground_truth() -> HashMap<String, GroundTruthEntry> {
    let mut gt = HashMap::new();

    // Nipah G (2VWD) - from m102.4 antibody studies
    gt.insert("2VWD".into(), GroundTruthEntry {
        cryptic_residues: vec![
            // Central cavity (m102.4 binding site)
            507, 508, 509, 510, 511, 529, 530, 531,
            // Dimeric interface (only accessible in tetramer)
            252, 253, 254, 255,
        ],
        epitope_residues: vec![
            // m102.4 epitope
            507, 508, 509, 510, 511, 529, 530, 531, 532,
        ],
        known_binders: vec!["m102.4".into()],
        source: "Xu K, et al. (2008) PNAS".into(),
    });

    gt
}
```

## Implementation Priority

1. **Week 1**: Biological assembly parsing + chain preservation
2. **Week 2**: Interface detection + cryptic score boost
3. **Week 3**: Functional site proximity weighting
4. **Week 4**: Antibody validation + Henipavirus benchmark

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| m102.4 epitope recall | N/A | ≥70% |
| Interface site detection | N/A | ≥80% |
| Henipavirus benchmark F1 | N/A | ≥0.65 |
| Cross-structure generalization | N/A | ≤10% drop |

## Files to Create

| File | Purpose |
|------|---------|
| `oligomer_topology.rs` | Biological assembly parsing |
| `interface_detector.rs` | Inter-chain interface detection |
| `functional_site_scorer.rs` | Proximity weighting |
| `antibody_validation.rs` | m102.4 and other antibody validation |
| `henipavirus_benchmark.rs` | Henipavirus-specific benchmarks |
