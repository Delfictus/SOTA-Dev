# PRISM4D Drug Discovery Platform — Implementation Blueprint

**Version**: 1.0
**Date**: 2026-02-11
**Scope**: Complete GPU-accelerated drug discovery pipeline
**Architecture**: Rust + CUDA + Python (RDKit/OpenFE)
**License Strategy**: Proprietary core + Apache/BSD/MIT open-source integrations

---

## TABLE OF CONTENTS

1. [Architecture Overview](#1-architecture-overview)
2. [Phase 0: Current State Inventory](#2-phase-0-current-state-inventory)
3. [Phase 1: Receptor & Ligand Preparation Engine](#3-phase-1-receptor--ligand-preparation-engine)
4. [Phase 2: UniDock GPU Docking Integration](#4-phase-2-unidock-gpu-docking-integration)
5. [Phase 3: GNINA CNN Rescoring](#5-phase-3-gnina-cnn-rescoring)
6. [Phase 4: Pharmacophore Model Generation](#6-phase-4-pharmacophore-model-generation)
7. [Phase 5: Virtual Screening Pipeline](#7-phase-5-virtual-screening-pipeline)
8. [Phase 6: Alchemical Free Energy (OpenFE)](#8-phase-6-alchemical-free-energy-openfe)
9. [Phase 7: Unified CLI & Orchestration](#9-phase-7-unified-cli--orchestration)
10. [Phase 8: Packaging & Distribution](#10-phase-8-packaging--distribution)
11. [Data Flow Specification](#11-data-flow-specification)
12. [File Manifest](#12-file-manifest)
13. [Dependency Matrix](#13-dependency-matrix)
14. [Testing Strategy](#14-testing-strategy)
15. [Risk Register](#15-risk-register)

---

## 1. Architecture Overview

### 1.1 Full Pipeline Data Flow

```
                        ┌──────────────────────┐
                        │   INPUT: PDB / CIF    │
                        │   + Ligand Library     │
                        └──────────┬─────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  PHASE 1: PREPARATION        │
                    │  ┌────────────────────────┐  │
                    │  │ Receptor Prep (Rust)    │  │
                    │  │ • PDB → Topology JSON   │  │
                    │  │ • Add H, assign charges │  │
                    │  │ • AMBER ff14SB params   │  │
                    │  │ • Generate PDBQT        │  │
                    │  └────────────────────────┘  │
                    │  ┌────────────────────────┐  │
                    │  │ Ligand Prep (RDKit)     │  │
                    │  │ • SMILES/SDF → 3D conf  │  │
                    │  │ • Protonation (pH 7.4)  │  │
                    │  │ • Gasteiger charges     │  │
                    │  │ • Generate PDBQT        │  │
                    │  └────────────────────────┘  │
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  PRISM4D CORE (Existing)     │
                    │  ┌────────────────────────┐  │
                    │  │ NHS RT Full Pipeline    │  │
                    │  │ • AMBER MD on GPU       │  │
                    │  │ • UV pump-probe spikes  │  │
                    │  │ • LIF neuromorphic det  │  │
                    │  │ • RT-core clustering    │  │
                    │  └────────────────────────┘  │
                    │  Output: binding_sites.json  │
                    │  Output: spike_events.json   │
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  PHASE 2: GPU DOCKING         │
                    │  ┌────────────────────────┐  │
                    │  │ UniDock (GPU Vina)      │  │
                    │  │ • Auto-gen docking box  │  │
                    │  │   from binding_sites    │  │
                    │  │ • Batch dock all sites  │  │
                    │  │ • 10K compounds/hr/GPU  │  │
                    │  │ • Output: ranked poses  │  │
                    │  └────────────────────────┘  │
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  PHASE 3: CNN RESCORING       │
                    │  ┌────────────────────────┐  │
                    │  │ GNINA (CNN Scoring)     │  │
                    │  │ • Rescore top N poses   │  │
                    │  │ • CNN affinity + pose   │  │
                    │  │ • Consensus w/ Vina     │  │
                    │  │ • Filter clashes        │  │
                    │  └────────────────────────┘  │
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  PHASE 4: PHARMACOPHORE       │
                    │  ┌────────────────────────┐  │
                    │  │ RDKit Pharmacophore     │  │
                    │  │ • Feature perception    │  │
                    │  │   (donor/acceptor/      │  │
                    │  │    hydrophobic/aromatic/ │  │
                    │  │    pos-ion/neg-ion)     │  │
                    │  │ • Spike-weighted model  │  │
                    │  │ • Distance constraints  │  │
                    │  │ • Directionality vecs   │  │
                    │  │ • Pharmacophore search  │  │
                    │  └────────────────────────┘  │
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  PHASE 5: VIRTUAL SCREENING   │
                    │  ┌────────────────────────┐  │
                    │  │ Multi-Stage Filter      │  │
                    │  │ Stage 1: Pharmacophore  │  │
                    │  │   match (shape + feat)  │  │
                    │  │ Stage 2: UniDock GPU    │  │
                    │  │   dock survivors        │  │
                    │  │ Stage 3: GNINA rescore  │  │
                    │  │ Stage 4: ADMET filter   │  │
                    │  │   (Lipinski/Veber/PAINS)│  │
                    │  │ Stage 5: Rank & report  │  │
                    │  └────────────────────────┘  │
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  PHASE 6: FREE ENERGY         │
                    │  ┌────────────────────────┐  │
                    │  │ OpenFE (Alchemical FEP) │  │
                    │  │ • RBFE for top hits     │  │
                    │  │ • ΔΔG predictions       │  │
                    │  │ • Lead optimization     │  │
                    │  │ • SAR analysis          │  │
                    │  └────────────────────────┘  │
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  OUTPUT: Evidence Pack        │
                    │  • Ranked compounds           │
                    │  • Binding poses (PDB)        │
                    │  • Affinity predictions        │
                    │  • Pharmacophore model         │
                    │  • FEP ΔΔG values             │
                    │  • PyMOL/ChimeraX sessions    │
                    │  • HTML/PDF report            │
                    └──────────────────────────────┘
```

### 1.2 Technology Stack

| Layer | Technology | Role |
|-------|-----------|------|
| Core engine | Rust + CUDA | MD simulation, spike detection, orchestration |
| GPU docking | UniDock (C++/CUDA) | Massively parallel molecular docking |
| CNN scoring | GNINA (C++/CUDA) | Deep learning pose rescoring |
| Cheminformatics | RDKit (C++/Python) | Pharmacophore, ADMET, molecular properties |
| Free energy | OpenFE (Python) | Alchemical binding free energy |
| Build system | Cargo + CMake + pip | Multi-language build |
| CLI | Clap (Rust) | Unified command-line interface |

### 1.3 Crate Layout (New + Existing)

```
crates/
├── prism-nhs/          # [EXISTS] Core spike detection
├── prism-gpu/          # [EXISTS] CUDA kernels
├── prism-report/       # [EXISTS] Evidence pack generation
├── prism-io/           # [EXISTS] I/O formats
├── prism-dock/         # [NEW] Docking orchestration (UniDock + GNINA)
├── prism-pharma/       # [NEW] Pharmacophore engine (RDKit FFI)
├── prism-screen/       # [NEW] Virtual screening pipeline
├── prism-fep/          # [NEW] Free energy perturbation (OpenFE bridge)
├── prism-prep/         # [NEW] Receptor + ligand preparation (Rust)
├── prism-chem/         # [NEW] Cheminformatics utilities (RDKit FFI)
└── prism-cli/          # [MODIFY] Unified CLI entry point
```

---

## 2. Phase 0: Current State Inventory

### 2.1 What Exists and Works

| Component | Status | Location | Notes |
|-----------|--------|----------|-------|
| NHS RT Full pipeline | VALIDATED | `crates/prism-nhs/src/bin/nhs_rt_full.rs` | 6/6 targets pass |
| Binding site JSON output | STABLE | `*.binding_sites.json` | Schema frozen |
| Spike events JSON output | STABLE | `*.site0.spike_events.json` | Per-site events |
| GPU pharmacophore splatting | COMPILED | `crates/prism-nhs/src/pharmacophore_gpu.rs` | Density maps only |
| CUDA kernel build system | STABLE | `crates/prism-gpu/build.rs` | sm_120, PTX output |
| Batch manifest system | STABLE | `batch_manifest_all14.json` | 14-structure batch |
| Evidence pack generator | STABLE | `crates/prism-report/src/finalize.rs` | HTML+PDF+PDB |
| Python prep scripts | STABLE | `scripts/stage1_sanitize.py`, `stage2_topology.py` | PDB→topology |
| Custom PDB→PDBQT converter | BASIC | `results_kras/pdb2pdbqt.py` | No charge calculation |
| Desktop docking script | BUGGY | `~/Desktop/prism4d_dock.py` | Overwrites configs |

### 2.2 What Does NOT Exist Yet

| Component | Priority | Complexity | Phase |
|-----------|----------|------------|-------|
| UniDock integration | CRITICAL | Medium | 2 |
| GNINA integration | HIGH | Medium | 3 |
| RDKit pharmacophore perception | HIGH | High | 4 |
| Proper ligand preparation (3D, charges, tautomers) | CRITICAL | Medium | 1 |
| Proper receptor preparation (H, charges, PDBQT) | CRITICAL | Medium | 1 |
| Virtual screening batch pipeline | HIGH | High | 5 |
| OpenFE bridge | MEDIUM | High | 6 |
| Unified CLI | HIGH | Medium | 7 |
| Packaging/distribution | MEDIUM | Medium | 8 |

---

## 3. Phase 1: Receptor & Ligand Preparation Engine

### 3.1 Objective

Replace the hacky `pdb2pdbqt.py` and ad-hoc ligand prep with a robust, reproducible preparation pipeline that handles:
- Receptor: PDB → protonated → AMBER charges → PDBQT (with correct AD4 atom types)
- Ligand: SMILES/SDF → 3D conformer → protonated → Gasteiger charges → PDBQT

### 3.2 New Crate: `prism-prep`

**Location**: `crates/prism-prep/`

```
crates/prism-prep/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── receptor.rs          # Receptor preparation
│   ├── ligand.rs            # Ligand preparation (calls RDKit via Python)
│   ├── pdbqt.rs             # PDBQT format writer
│   ├── protonation.rs       # pH-dependent protonation
│   ├── atom_typing.rs       # AutoDock4 atom type assignment
│   ├── charges.rs           # Gasteiger charge calculation
│   └── bin/
│       └── prism_prep.rs    # CLI binary
├── python/
│   └── ligand_prep.py       # RDKit-based ligand preparation
└── tests/
    ├── test_receptor.rs
    └── test_ligand.rs
```

### 3.3 Receptor Preparation — `receptor.rs`

**Input**: PDB file (crystal structure or Prism4D open-frame PDB)
**Output**: `receptor.pdbqt` with correct atom types and charges

#### Data Structures

```rust
/// Atom with full preparation metadata.
pub struct PreparedAtom {
    pub serial: u32,
    pub name: String,           // e.g. "CA", "CB", "OG1"
    pub residue_name: String,   // e.g. "SER", "ALA"
    pub chain: char,
    pub resid: i32,
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub occupancy: f64,
    pub bfactor: f64,
    pub element: String,        // "C", "N", "O", "S", "H"
    pub charge: f64,            // Gasteiger partial charge
    pub ad4_type: Ad4AtomType,  // AutoDock4 atom type
    pub is_polar_h: bool,       // HD classification
}

/// AutoDock4 atom types (exhaustive).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Ad4AtomType {
    C,   // Non-aromatic carbon
    A,   // Aromatic carbon
    N,   // Non-H-bond-accepting nitrogen
    NA,  // H-bond accepting nitrogen (e.g. His ND1 unprotonated)
    NS,  // Amide nitrogen
    OA,  // H-bond accepting oxygen (carbonyl O, carboxylate O, ether O)
    OS,  // Sulfate oxygen
    SA,  // H-bond accepting sulfur (e.g. Met SD)
    S,   // Non-accepting sulfur (e.g. Cys SG in disulfide)
    HD,  // H-bond donating hydrogen (on N-H, O-H, S-H)
    H,   // Non-polar hydrogen
    F,   // Fluorine
    Cl,  // Chlorine
    Br,  // Bromine
    I,   // Iodine
    P,   // Phosphorus
    Fe,  // Iron
    Zn,  // Zinc
    Mg,  // Magnesium
    Ca,  // Calcium
    Mn,  // Manganese
}

/// Full prepared receptor.
pub struct PreparedReceptor {
    pub atoms: Vec<PreparedAtom>,
    pub source_pdb: PathBuf,
    pub n_residues: usize,
    pub n_chains: usize,
    pub has_hydrogens: bool,
    pub charge_method: String,   // "gasteiger" or "amber"
}
```

#### Key Functions

```rust
/// Load PDB, add hydrogens if missing, assign charges and AD4 types.
pub fn prepare_receptor(
    pdb_path: &Path,
    options: &ReceptorPrepOptions,
) -> anyhow::Result<PreparedReceptor>;

/// Options for receptor preparation.
pub struct ReceptorPrepOptions {
    pub add_hydrogens: bool,          // Default: true
    pub ph: f64,                      // Default: 7.4
    pub remove_water: bool,           // Default: true
    pub remove_heteroatoms: bool,     // Default: false (keep cofactors)
    pub keep_chains: Option<Vec<char>>, // None = all chains
    pub charge_method: ChargeMethod,  // Gasteiger or AMBER
}

/// Assign AD4 atom type based on element, residue context, and bonding.
pub fn assign_ad4_type(
    atom: &PdbAtom,
    residue: &Residue,
    neighbors: &[&PdbAtom],
) -> Ad4AtomType;

/// Atom type rules (exhaustive, no ambiguity):
///
/// CARBON:
///   - In aromatic ring (PHE CE1/CE2/CZ, TYR CE1/CE2/CZ, TRP CE2/CE3/CZ2/CZ3/CH2,
///     HIS CD2/CE1) → A (aromatic)
///   - All other C → C
///
/// NITROGEN:
///   - Backbone N (with attached H) → N
///   - LYS NZ (protonated amine, +charge) → N
///   - ARG NH1/NH2/NE → N
///   - HIS ND1 (protonated, has H) → N
///   - HIS NE2 (unprotonated, lone pair) → NA (acceptor)
///   - ASN ND2, GLN NE2 (amide) → N
///   - PRO N (no H, tertiary amine) → NA
///
/// OXYGEN:
///   - Backbone O (carbonyl) → OA (acceptor)
///   - ASP OD1/OD2, GLU OE1/OE2 (carboxylate) → OA
///   - SER OG, THR OG1 (hydroxyl) → OA
///   - TYR OH → OA
///   - ASN OD1, GLN OE1 (amide carbonyl) → OA
///   - Water O → OA
///
/// SULFUR:
///   - MET SD (thioether, lone pair) → SA (acceptor)
///   - CYS SG (free thiol) → SA
///   - CYS SG (disulfide bond) → S
///
/// HYDROGEN:
///   - Bonded to N, O, or S → HD (polar, H-bond donor)
///   - Bonded to C → H (non-polar)
///
/// HALIDES:
///   - F → F, Cl → Cl, Br → Br, I → I
///
/// METALS:
///   - Fe → Fe, Zn → Zn, Mg → Mg, Ca → Ca, Mn → Mn, P → P

/// Write PDBQT file.
pub fn write_pdbqt(receptor: &PreparedReceptor, path: &Path) -> anyhow::Result<()>;
```

#### Atom Typing Decision Table (Complete)

```
┌──────────────┬──────────────┬──────────┬───────────────────────────────────┐
│ Residue      │ Atom Name    │ AD4 Type │ Rationale                         │
├──────────────┼──────────────┼──────────┼───────────────────────────────────┤
│ ALL          │ N (backbone) │ N        │ Amide N, has H (except PRO)       │
│ PRO          │ N            │ NA       │ Tertiary amine, no H, acceptor    │
│ ALL          │ CA           │ C        │ Alpha carbon, non-aromatic        │
│ ALL          │ C (backbone) │ C        │ Carbonyl carbon                   │
│ ALL          │ O (backbone) │ OA       │ Carbonyl oxygen, H-bond acceptor  │
│ ALL          │ H (on N)     │ HD       │ Polar H, donor                    │
│ ALL          │ HA, HB*, HG* │ H        │ Non-polar H on carbon             │
│              │ HD*, HE*     │          │                                   │
├──────────────┼──────────────┼──────────┼───────────────────────────────────┤
│ ALA          │ CB           │ C        │ Methyl carbon                     │
│ ARG          │ CZ           │ C        │ Guanidinium carbon                │
│ ARG          │ NH1, NH2     │ N        │ Guanidinium N, donors             │
│ ARG          │ NE           │ N        │ Guanidinium N, donor              │
│ ARG          │ HH*, HE     │ HD       │ Polar H on N                      │
│ ASN          │ OD1          │ OA       │ Amide carbonyl, acceptor          │
│ ASN          │ ND2          │ N        │ Amide nitrogen, donor             │
│ ASN          │ HD21, HD22   │ HD       │ Polar H on amide N                │
│ ASP          │ OD1, OD2     │ OA       │ Carboxylate O, acceptor           │
│ CYS (free)   │ SG           │ SA       │ Free thiol, acceptor              │
│ CYS (SS)     │ SG           │ S        │ Disulfide, non-acceptor           │
│ CYS          │ HG           │ HD       │ Thiol H (if present)              │
│ GLN          │ OE1          │ OA       │ Amide carbonyl, acceptor          │
│ GLN          │ NE2          │ N        │ Amide N, donor                    │
│ GLU          │ OE1, OE2     │ OA       │ Carboxylate O, acceptor           │
│ GLY          │ (no sidechain) │ —      │                                   │
│ HIS (HIE)    │ ND1          │ NA       │ Unprotonated, acceptor            │
│ HIS (HIE)    │ NE2          │ N        │ Protonated, donor                 │
│ HIS (HID)    │ ND1          │ N        │ Protonated, donor                 │
│ HIS (HID)    │ NE2          │ NA       │ Unprotonated, acceptor            │
│ HIS (HIP)    │ ND1, NE2     │ N        │ Both protonated, donors           │
│ HIS          │ CD2, CE1     │ A        │ Aromatic carbons in imidazole     │
│ HIS          │ CG           │ A        │ Aromatic carbon                   │
│ ILE          │ all C        │ C        │ Aliphatic carbons                 │
│ LEU          │ all C        │ C        │ Aliphatic carbons                 │
│ LYS          │ NZ           │ N        │ Protonated amine, donor           │
│ LYS          │ HZ1-3        │ HD       │ Polar H on amine                  │
│ MET          │ SD           │ SA       │ Thioether S, acceptor             │
│ PHE          │ CG, CD1, CD2 │ A        │ Aromatic ring carbons             │
│ PHE          │ CE1, CE2, CZ │ A        │ Aromatic ring carbons             │
│ SER          │ OG           │ OA       │ Hydroxyl O, acceptor              │
│ SER          │ HG           │ HD       │ Hydroxyl H, donor                 │
│ THR          │ OG1          │ OA       │ Hydroxyl O, acceptor              │
│ THR          │ HG1          │ HD       │ Hydroxyl H, donor                 │
│ TRP          │ NE1          │ N        │ Indole NH, donor                  │
│ TRP          │ HE1          │ HD       │ Indole H, donor                   │
│ TRP          │ CD1, CE2     │ A        │ Aromatic carbons                  │
│ TRP          │ CE3, CZ2     │ A        │ Aromatic carbons                  │
│ TRP          │ CZ3, CH2     │ A        │ Aromatic carbons                  │
│ TRP          │ CD2, CG      │ A        │ Aromatic carbons (ring junction)  │
│ TYR          │ CG, CD1, CD2 │ A        │ Aromatic ring carbons             │
│ TYR          │ CE1, CE2, CZ │ A        │ Aromatic ring carbons             │
│ TYR          │ OH           │ OA       │ Phenol O, acceptor                │
│ TYR          │ HH           │ HD       │ Phenol H, donor                   │
│ VAL          │ all C        │ C        │ Aliphatic carbons                 │
└──────────────┴──────────────┴──────────┴───────────────────────────────────┘
```

### 3.4 Ligand Preparation — `ligand.rs` + `python/ligand_prep.py`

**Input**: SMILES string, SDF file, or PDB/MOL2
**Output**: 3D PDBQT with Gasteiger charges and rotatable bond tree

#### Python Worker (called from Rust via subprocess)

**File**: `crates/prism-prep/python/ligand_prep.py`

```python
#!/usr/bin/env python3
"""
PRISM4D Ligand Preparation Worker
Called by Rust orchestrator. Reads JSON config from stdin, writes PDBQT to stdout.

Dependencies: rdkit, meeko
"""

# Input JSON schema:
# {
#   "input_type": "smiles" | "sdf" | "pdb",
#   "input_data": "<SMILES string>" | "<path to file>",
#   "name": "sotorasib",
#   "ph": 7.4,
#   "num_conformers": 1,
#   "minimize": true,
#   "output_pdbqt": "/path/to/output.pdbqt"
# }
#
# Pipeline:
# 1. Parse input (SMILES → mol, SDF → mol, PDB → mol)
# 2. Add hydrogens at specified pH
# 3. Generate 3D conformer (ETKDG v3)
# 4. Minimize with MMFF94s (200 steps)
# 5. Compute Gasteiger charges
# 6. Convert to PDBQT via meeko (MoleculePreparation)
# 7. Write PDBQT to output path
#
# Output JSON schema (to stdout):
# {
#   "status": "success" | "error",
#   "output_pdbqt": "/path/to/output.pdbqt",
#   "num_atoms": 42,
#   "num_rotatable_bonds": 6,
#   "molecular_weight": 560.6,
#   "logp": 3.2,
#   "hbd": 2,
#   "hba": 7,
#   "error": null
# }
```

#### RDKit Processing Steps (Exact)

```
Step 1: Parse molecule
  - SMILES: Chem.MolFromSmiles(smiles)
  - SDF: Chem.SDMolSupplier(path)[0]
  - PDB: Chem.MolFromPDBFile(path)
  → Fail if None (invalid input)

Step 2: Sanitize
  - Chem.SanitizeMol(mol)
  - Remove salts: rdMolStandardize.FragmentParent(mol)
  - Normalize: rdMolStandardize.Normalize(mol)

Step 3: Protonate at pH
  - Use rdMolStandardize.Uncharger() to neutralize
  - Then reprotonate: apply pKa rules
    - Carboxylic acid (pKa ~4.5): deprotonate at pH 7.4 → COO⁻
    - Primary amine (pKa ~10): protonate at pH 7.4 → NH₃⁺
    - Imidazole (pKa ~6.0): 50/50 at pH 7.4 → leave neutral
    - Guanidinium (pKa ~12.5): protonated → +1
    - Phosphate (pKa ~2.1): deprotonate → PO₄²⁻

Step 4: Add explicit hydrogens
  - mol = Chem.AddHs(mol)

Step 5: Generate 3D conformer
  - AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
  - If fails: AllChem.EmbedMolecule(mol, AllChem.ETKDG())
  - If fails: AllChem.EmbedMolecule(mol, useRandomCoords=True)
  → Fail if all embedding fails

Step 6: Energy minimize
  - result = AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
  - If MMFF fails: AllChem.UFFOptimizeMolecule(mol, maxIters=200)

Step 7: Gasteiger charges
  - AllChem.ComputeGasteigerCharges(mol)

Step 8: Convert to PDBQT via meeko
  - from meeko import MoleculePreparation, PDBQTWriterLegacy
  - preparator = MoleculePreparation()
  - mol_setup = preparator.prepare(mol)[0]
  - pdbqt_string = PDBQTWriterLegacy.write_string(mol_setup)[0]
  - Write to output path

Step 9: Compute molecular properties
  - MW = Descriptors.MolWt(mol)
  - LogP = Descriptors.MolLogP(mol)
  - HBD = Descriptors.NumHDonors(mol)
  - HBA = Descriptors.NumHAcceptors(mol)
  - RotBonds = Descriptors.NumRotatableBonds(mol)
  - TPSA = Descriptors.TPSA(mol)
```

#### Rust Orchestrator — `ligand.rs`

```rust
/// Prepared ligand with full metadata.
pub struct PreparedLigand {
    pub name: String,
    pub smiles: String,
    pub pdbqt_path: PathBuf,
    pub num_atoms: u32,
    pub num_rotatable_bonds: u32,
    pub molecular_weight: f64,
    pub logp: f64,
    pub hbd: u32,              // H-bond donors
    pub hba: u32,              // H-bond acceptors
    pub tpsa: f64,             // Topological polar surface area
    pub lipinski_violations: u32,
}

/// Prepare a single ligand.
pub fn prepare_ligand(
    input: LigandInput,
    output_dir: &Path,
    options: &LigandPrepOptions,
) -> anyhow::Result<PreparedLigand>;

/// Batch prepare ligands (parallel via rayon).
pub fn prepare_ligand_batch(
    inputs: &[LigandInput],
    output_dir: &Path,
    options: &LigandPrepOptions,
    max_parallel: usize,       // Default: num_cpus
) -> anyhow::Result<Vec<PreparedLigand>>;

pub enum LigandInput {
    Smiles { smiles: String, name: String },
    SdfFile { path: PathBuf },
    SdfBatch { path: PathBuf },       // Multi-molecule SDF
    PdbFile { path: PathBuf, name: String },
}

pub struct LigandPrepOptions {
    pub ph: f64,                  // Default: 7.4
    pub num_conformers: u32,      // Default: 1
    pub minimize: bool,           // Default: true
    pub compute_properties: bool, // Default: true
}
```

### 3.5 Docking Box Generation — `docking_box.rs`

**Purpose**: Auto-generate docking boxes from `binding_sites.json`

```rust
/// Docking box specification.
pub struct DockingBox {
    pub site_id: u32,
    pub center: [f64; 3],     // From binding_sites.json centroid
    pub size: [f64; 3],       // From bounding_box + padding
    pub exhaustiveness: u32,
    pub num_modes: u32,
    pub energy_range: f64,
}

/// Generate docking boxes for all sites in binding_sites.json.
pub fn generate_docking_boxes(
    binding_sites_path: &Path,
    options: &DockingBoxOptions,
) -> anyhow::Result<Vec<DockingBox>>;

pub struct DockingBoxOptions {
    pub padding: f64,             // Extra Å around bounding box. Default: 5.0
    pub min_box_size: f64,        // Minimum box dimension. Default: 20.0
    pub max_box_size: f64,        // Maximum box dimension. Default: 30.0
    pub exhaustiveness: u32,      // Default: 32
    pub num_modes: u32,           // Default: 20
    pub energy_range: f64,        // Default: 5.0 kcal/mol
    pub druggable_only: bool,     // Only generate boxes for druggable sites. Default: true
    pub min_druggability: f64,    // Minimum druggability score. Default: 0.3
}

/// Write Vina-format config file.
pub fn write_vina_config(
    dbox: &DockingBox,
    receptor_pdbqt: &str,
    output_path: &Path,
) -> anyhow::Result<()>;
```

### 3.6 Success Criteria — Phase 1

- [ ] `prism-prep receptor -i 4obe.pdb -o receptor.pdbqt` produces PDBQT with correct AD4 types
- [ ] All 20 standard amino acids correctly typed (verified against MGLTools output)
- [ ] Aromatic residues (PHE, TYR, TRP, HIS) get `A` type for ring carbons
- [ ] Polar H atoms get `HD`, non-polar get `H`
- [ ] `prism-prep ligand -s "CC(=O)Nc1ccc(O)cc1" -o acetaminophen.pdbqt` produces valid 3D PDBQT
- [ ] Batch ligand prep: 1000 SMILES → PDBQT in < 5 minutes
- [ ] Docking boxes auto-generated from any `binding_sites.json`
- [ ] Round-trip test: prep receptor + ligand, dock with Vina, get negative affinity

---

## 4. Phase 2: UniDock GPU Docking Integration

### 4.1 Objective

Integrate UniDock as the primary docking engine, callable from Rust, with automatic parallelization across detected binding sites.

### 4.2 UniDock Overview

- **What**: GPU-accelerated Vina fork by DP Technology
- **Speed**: ~1000x faster than CPU Vina (10,000+ compounds/hour on single GPU)
- **Compatibility**: Same input format as Vina (PDBQT + config)
- **License**: Apache 2.0
- **Repository**: https://github.com/dptech-corp/Uni-Dock

### 4.3 Installation Strategy

```bash
# Option A: Pre-built binary (preferred)
# Download from GitHub releases, place in $PRISM4D_HOME/bin/unidock

# Option B: Build from source
git clone https://github.com/dptech-corp/Uni-Dock.git
cd Uni-Dock/unidock
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
# Binary: build/unidock
```

### 4.4 New Crate: `prism-dock`

**Location**: `crates/prism-dock/`

```
crates/prism-dock/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── engine.rs            # Docking engine abstraction
│   ├── unidock.rs           # UniDock backend
│   ├── vina.rs              # Vina backend (fallback)
│   ├── config.rs            # Docking configuration
│   ├── results.rs           # Docking results parsing
│   ├── pose.rs              # Pose data structures
│   ├── batch.rs             # Batch docking orchestration
│   └── bin/
│       └── prism_dock.rs    # CLI binary
└── tests/
    ├── test_unidock.rs
    └── test_results.rs
```

#### Data Structures

```rust
/// Docking engine trait — backend-agnostic.
pub trait DockingEngine {
    fn name(&self) -> &str;
    fn dock_single(
        &self,
        receptor: &Path,
        ligand: &Path,
        config: &DockingConfig,
        output: &Path,
    ) -> anyhow::Result<DockingResult>;

    fn dock_batch(
        &self,
        receptor: &Path,
        ligands: &[PathBuf],
        config: &DockingConfig,
        output_dir: &Path,
    ) -> anyhow::Result<Vec<DockingResult>>;

    fn supports_gpu_batch(&self) -> bool;
}

/// UniDock-specific engine.
pub struct UniDockEngine {
    binary_path: PathBuf,          // Path to unidock executable
    gpu_device: u32,               // CUDA device ID
    batch_size: u32,               // Ligands per GPU batch (default: 20)
}

/// Docking configuration.
pub struct DockingConfig {
    pub center: [f64; 3],
    pub size: [f64; 3],
    pub exhaustiveness: u32,        // Default: 32
    pub num_modes: u32,             // Default: 20
    pub energy_range: f64,          // Default: 5.0 kcal/mol
    pub seed: Option<i64>,          // Reproducibility
    pub num_cpus: Option<u32>,      // CPU threads (Vina only)
}

/// Single docking result.
pub struct DockingResult {
    pub ligand_name: String,
    pub site_id: u32,
    pub poses: Vec<DockingPose>,
    pub runtime_seconds: f64,
    pub engine: String,            // "unidock" or "vina"
}

/// A single docking pose.
pub struct DockingPose {
    pub rank: u32,
    pub affinity_kcal: f64,        // Vina/UniDock affinity score
    pub rmsd_lb: f64,              // RMSD lower bound from best
    pub rmsd_ub: f64,              // RMSD upper bound from best
    pub atoms: Vec<PoseAtom>,      // 3D coordinates of docked pose
}

pub struct PoseAtom {
    pub name: String,
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub ad4_type: String,
    pub charge: f64,
}

/// Batch docking job specification.
pub struct BatchDockJob {
    pub receptor_pdbqt: PathBuf,
    pub ligands: Vec<PathBuf>,           // List of PDBQT files
    pub sites: Vec<DockingBox>,          // From Phase 1 box generation
    pub output_dir: PathBuf,
    pub engine: DockingEngineType,
    pub max_parallel_sites: usize,       // Default: 1 (sequential sites)
    pub top_n_per_site: usize,           // Keep top N ligands per site. Default: all
}

pub enum DockingEngineType {
    UniDock,       // Preferred
    Vina,          // Fallback
    Auto,          // UniDock if available, else Vina
}
```

#### UniDock Invocation — `unidock.rs`

```rust
impl DockingEngine for UniDockEngine {
    fn dock_batch(
        &self,
        receptor: &Path,
        ligands: &[PathBuf],
        config: &DockingConfig,
        output_dir: &Path,
    ) -> anyhow::Result<Vec<DockingResult>> {
        // UniDock supports batch mode natively:
        // unidock --receptor receptor.pdbqt \
        //         --gpu_batch_ligand_dir ligands/ \
        //         --center_x X --center_y Y --center_z Z \
        //         --size_x SX --size_y SY --size_z SZ \
        //         --dir output/ \
        //         --exhaustiveness 32 \
        //         --num_modes 20

        // 1. Create ligand batch directory (symlinks)
        // 2. Build command
        // 3. Execute subprocess
        // 4. Parse output PDBQT files
        // 5. Return sorted results
    }
}
```

#### PDBQT Result Parser — `results.rs`

```rust
/// Parse docked PDBQT output file.
/// Extracts all MODEL blocks with REMARK VINA RESULT scores.
pub fn parse_docked_pdbqt(path: &Path) -> anyhow::Result<Vec<DockingPose>>;

/// Aggregate results across sites and ligands.
pub fn aggregate_results(
    results: &[DockingResult],
) -> DockingSummary;

pub struct DockingSummary {
    pub total_ligands: u32,
    pub total_poses: u32,
    pub sites_docked: u32,
    pub best_per_site: Vec<SiteBestResult>,
    pub best_overall: Option<DockingPose>,
    pub runtime_total_seconds: f64,
}

pub struct SiteBestResult {
    pub site_id: u32,
    pub centroid: [f64; 3],
    pub best_ligand: String,
    pub best_affinity: f64,
    pub num_ligands_docked: u32,
}
```

### 4.5 Output Format — `docking_results.json`

```json
{
  "prism4d_version": "0.4.0",
  "engine": "unidock",
  "receptor": "4obe_mono.pdbqt",
  "timestamp": "2026-02-11T15:30:00Z",
  "runtime_seconds": 45.2,
  "sites": [
    {
      "site_id": 0,
      "centroid": [-5.425, -22.45, 41.841],
      "box_size": [25.0, 25.0, 25.0],
      "ligands": [
        {
          "name": "sotorasib",
          "poses": [
            {
              "rank": 1,
              "affinity_kcal": -8.7,
              "rmsd_lb": 0.0,
              "rmsd_ub": 0.0,
              "pdbqt_file": "ligands/sotorasib_site0_docked.pdbqt"
            },
            {
              "rank": 2,
              "affinity_kcal": -8.2,
              "rmsd_lb": 2.1,
              "rmsd_ub": 4.3
            }
          ]
        }
      ]
    }
  ]
}
```

### 4.6 Success Criteria — Phase 2

- [ ] `prism-dock --receptor receptor.pdbqt --ligand sotorasib.pdbqt --config site0.txt` produces valid docked output
- [ ] Batch mode: 100 ligands docked in < 10 minutes on single GPU
- [ ] Auto-generates docking boxes from `binding_sites.json`
- [ ] Falls back to Vina if UniDock not installed
- [ ] Results JSON matches schema above
- [ ] Known drug-target pair (e.g. sotorasib + KRAS G12C) scores < -7 kcal/mol

---

## 5. Phase 3: GNINA CNN Rescoring

### 5.1 Objective

Rescore UniDock poses with GNINA's convolutional neural network scoring function for improved ranking accuracy. GNINA's CNN outperforms Vina's empirical scoring on pose prediction benchmarks.

### 5.2 GNINA Overview

- **What**: Deep learning molecular docking (CNN scoring function)
- **Model**: 3D CNN trained on PDBbind crystal structures
- **Scores**: `CNNscore` (pose quality 0-1), `CNNaffinity` (predicted pKd)
- **License**: Apache 2.0 (choose Apache in dual-license)
- **Repository**: https://github.com/gnina/gnina

### 5.3 Integration in `prism-dock`

```
crates/prism-dock/src/
├── gnina.rs              # [NEW] GNINA backend
├── consensus.rs          # [NEW] Consensus scoring (Vina + GNINA)
```

#### Data Structures

```rust
/// GNINA scoring engine.
pub struct GninaEngine {
    binary_path: PathBuf,
    cnn_model: GninaCnnModel,
    gpu_device: u32,
}

pub enum GninaCnnModel {
    Default,                    // Built-in default CNN
    Dense,                      // dense model (more accurate, slower)
    CrossDocked2020,            // Trained on CrossDocked2020 set
    Custom(PathBuf),            // User-provided .model file
}

/// GNINA rescore result.
pub struct GninaScore {
    pub cnn_score: f64,         // 0.0-1.0, pose quality prediction
    pub cnn_affinity: f64,      // Predicted -log(Kd), higher = better
    pub vina_affinity: f64,     // Original Vina/UniDock score
}

/// Consensus score combining Vina + GNINA.
pub struct ConsensusScore {
    pub ligand_name: String,
    pub site_id: u32,
    pub pose_rank: u32,
    pub vina_affinity: f64,     // kcal/mol (negative = better)
    pub cnn_score: f64,         // 0-1 (higher = better)
    pub cnn_affinity: f64,      // predicted pKd (higher = better)
    pub consensus_rank: f64,    // Weighted combination
    pub pose_pdbqt: PathBuf,
}

/// Consensus ranking formula:
///   consensus_rank = 0.4 * normalize(vina_affinity)
///                  + 0.3 * cnn_score
///                  + 0.3 * normalize(cnn_affinity)
///
/// Where normalize() maps to [0, 1] range across all poses.
///
/// Rationale:
///   - Vina captures physics-based interactions
///   - CNN_score captures geometric pose quality
///   - CNN_affinity captures learned binding affinity patterns
///   - Equal-ish weighting avoids over-reliance on any single method
```

#### GNINA Invocation

```rust
impl GninaEngine {
    /// Rescore existing poses (no re-docking).
    pub fn rescore(
        &self,
        receptor: &Path,
        docked_poses: &Path,     // PDBQT with multiple MODELs
        output: &Path,
    ) -> anyhow::Result<Vec<GninaScore>> {
        // gnina --receptor receptor.pdbqt \
        //       --ligand docked.pdbqt \
        //       --score_only \
        //       --cnn crossdock_default2018 \
        //       --out scored.pdbqt
        //
        // Parse SDF/PDBQT output for CNN scores
    }

    /// Full dock + score (alternative to UniDock).
    pub fn dock(
        &self,
        receptor: &Path,
        ligand: &Path,
        config: &DockingConfig,
        output: &Path,
    ) -> anyhow::Result<Vec<GninaScore>> {
        // gnina --receptor receptor.pdbqt \
        //       --ligand ligand.pdbqt \
        //       --center_x X --center_y Y --center_z Z \
        //       --size_x SX --size_y SY --size_z SZ \
        //       --exhaustiveness 32 \
        //       --cnn crossdock_default2018 \
        //       --out docked.pdbqt
    }
}
```

### 5.4 Success Criteria — Phase 3

- [ ] `prism-dock rescore --receptor r.pdbqt --poses docked.pdbqt` returns CNN scores
- [ ] Consensus ranking reorders poses (CNN sometimes disagrees with Vina)
- [ ] Known good poses get CNN_score > 0.7
- [ ] Known bad poses (decoys) get CNN_score < 0.3
- [ ] Batch rescore: 1000 poses in < 5 minutes on GPU

---

## 6. Phase 4: Pharmacophore Model Generation

### 6.1 Objective

Generate proper pharmacophore models with typed features (donor, acceptor, hydrophobic, aromatic, positive, negative) that can be used for:
1. Virtual screening (pharmacophore search against compound databases)
2. Scaffold hopping (find chemically different molecules with same pharmacophore)
3. Lead optimization (maintain critical interactions while modifying scaffold)

### 6.2 New Crate: `prism-pharma`

**Location**: `crates/prism-pharma/`

```
crates/prism-pharma/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── model.rs             # Pharmacophore model data structures
│   ├── perception.rs        # Feature perception (calls RDKit)
│   ├── spike_features.rs    # Map spike types to pharmacophore features
│   ├── alignment.rs         # Pharmacophore alignment scoring
│   ├── search.rs            # Pharmacophore database search
│   ├── io.rs                # Read/write pharmacophore formats
│   └── bin/
│       └── prism_pharma.rs  # CLI binary
├── python/
│   ├── pharmacophore_perception.py   # RDKit feature perception
│   ├── pharmacophore_search.py       # Database search
│   └── pharmacophore_align.py        # Alignment scoring
└── tests/
    └── test_pharmacophore.rs
```

### 6.3 Data Structures

```rust
/// A pharmacophore feature (point in 3D space with type and properties).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PharmacophoreFeature {
    pub feature_type: FeatureType,
    pub position: [f64; 3],           // 3D center of feature (Å)
    pub radius: f64,                  // Tolerance sphere radius (Å), default 1.5
    pub direction: Option<[f64; 3]>,  // Unit vector for directional features (HBD/HBA)
    pub weight: f64,                  // Importance weight (from spike density), default 1.0
    pub source: FeatureSource,        // Where this feature came from
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum FeatureType {
    HBondDonor,        // N-H, O-H, S-H pointing into pocket
    HBondAcceptor,     // Lone pair on N, O, S accessible in pocket
    Hydrophobic,       // Aliphatic/aromatic surface in pocket
    Aromatic,          // Pi system (ring centroid + normal vector)
    PositiveIonizable, // Cationic group (Lys NH3+, Arg guanidinium)
    NegativeIonizable, // Anionic group (Asp COO-, Glu COO-)
    Exclusion,         // Steric clash zone (receptor atoms)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeatureSource {
    /// From Prism4D spike events (weighted by spike density).
    SpikeCluster {
        spike_type: String,    // "TRP", "TYR", "PHE", "BNZ", "CATION", "ANION"
        spike_count: u32,      // Number of spikes contributing
        density_max: f64,      // Peak density value
    },
    /// From RDKit perception of docked ligand pose.
    LigandPose {
        ligand_name: String,
        atom_indices: Vec<u32>,
    },
    /// From receptor lining residue analysis.
    ReceptorSurface {
        residue: String,       // e.g. "ASP-184-A"
        atom_names: Vec<String>,
    },
}

/// Complete pharmacophore model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PharmacophoreModel {
    pub name: String,
    pub site_id: u32,
    pub centroid: [f64; 3],
    pub features: Vec<PharmacophoreFeature>,
    pub distance_constraints: Vec<DistanceConstraint>,
    pub excluded_volumes: Vec<ExcludedVolume>,
    pub metadata: PharmacophoreMetadata,
}

/// Distance constraint between two features.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistanceConstraint {
    pub feature_i: usize,     // Index into features vec
    pub feature_j: usize,
    pub distance: f64,        // Å
    pub tolerance: f64,       // ± Å (default 1.0)
}

/// Excluded volume (receptor clash zone).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExcludedVolume {
    pub position: [f64; 3],
    pub radius: f64,          // Å
}

/// Model metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PharmacophoreMetadata {
    pub receptor_pdb: String,
    pub prism4d_version: String,
    pub generation_method: String,  // "spike_density" | "ligand_pose" | "hybrid"
    pub spike_events_file: Option<String>,
    pub docked_pose_file: Option<String>,
    pub num_spikes_total: u32,
    pub timestamp: String,
}
```

### 6.4 Spike-to-Feature Mapping

```rust
/// Map Prism4D spike types to pharmacophore features.
///
/// Spike events record WHERE aromatic probes detected water exclusion.
/// This tells us WHAT KIND of interaction the pocket expects:
///
/// ┌────────────────┬──────────────────────┬─────────────────────────────────┐
/// │ Spike Type     │ Pharmacophore Feature │ Rationale                       │
/// ├────────────────┼──────────────────────┼─────────────────────────────────┤
/// │ TRP (280nm)    │ Hydrophobic +        │ Trp is large hydrophobic probe; │
/// │                │ Aromatic             │ high density = deep hydrophobic │
/// │                │                      │ pocket or pi-stacking site      │
/// ├────────────────┼──────────────────────┼─────────────────────────────────┤
/// │ TYR (274nm)    │ HBondDonor +         │ Tyr has OH (donor/acceptor) +   │
/// │                │ HBondAcceptor +      │ aromatic ring; density marks    │
/// │                │ Aromatic             │ mixed polar/aromatic sites      │
/// ├────────────────┼──────────────────────┼─────────────────────────────────┤
/// │ PHE (258nm)    │ Hydrophobic +        │ Phe is purely hydrophobic       │
/// │                │ Aromatic             │ aromatic; marks pi-stacking     │
/// │                │                      │ and van der Waals sites         │
/// ├────────────────┼──────────────────────┼─────────────────────────────────┤
/// │ BNZ (254nm)    │ Aromatic             │ Benzene virtual cosolvent;      │
/// │                │                      │ pure aromatic interaction probe │
/// ├────────────────┼──────────────────────┼─────────────────────────────────┤
/// │ SS (250nm)     │ HBondAcceptor        │ Disulfide probe marks S-rich    │
/// │                │                      │ environments; Met/Cys acceptors │
/// ├────────────────┼──────────────────────┼─────────────────────────────────┤
/// │ CATION         │ NegativeIonizable    │ Cation probe attracted =        │
/// │                │                      │ pocket has negative charge      │
/// │                │                      │ (Asp/Glu). Ligand needs cation. │
/// ├────────────────┼──────────────────────┼─────────────────────────────────┤
/// │ ANION          │ PositiveIonizable    │ Anion probe attracted =         │
/// │                │                      │ pocket has positive charge      │
/// │                │                      │ (Lys/Arg). Ligand needs anion.  │
/// └────────────────┴──────────────────────┴─────────────────────────────────┘
///
/// IMPORTANT: The mapping is INVERTED for charged features.
/// If a CATION probe is attracted to the pocket, it means the pocket
/// is NEGATIVE, so the LIGAND needs a POSITIVE group there → NegativeIonizable
/// Wait, no. Re-read:
///   - CATION spike = cationic probe detected water exclusion here
///   - This means the pocket has negative electrostatic potential
///   - A drug molecule binding here needs a POSITIVE ionizable group
///     to complement the pocket's negative charge
///   - So CATION spikes → PositiveIonizable feature for the LIGAND
///
/// CORRECTION:
/// │ CATION         │ PositiveIonizable    │ Cation attracted to neg pocket; │
/// │                │                      │ ligand needs + group here       │
/// │ ANION          │ NegativeIonizable    │ Anion attracted to pos pocket;  │
/// │                │                      │ ligand needs - group here       │

pub fn spike_type_to_features(spike_type: &str) -> Vec<FeatureType> {
    match spike_type {
        "TRP" => vec![FeatureType::Hydrophobic, FeatureType::Aromatic],
        "TYR" => vec![FeatureType::HBondDonor, FeatureType::HBondAcceptor, FeatureType::Aromatic],
        "PHE" => vec![FeatureType::Hydrophobic, FeatureType::Aromatic],
        "BNZ" => vec![FeatureType::Aromatic],
        "SS"  => vec![FeatureType::HBondAcceptor],
        "CATION" => vec![FeatureType::PositiveIonizable],
        "ANION"  => vec![FeatureType::NegativeIonizable],
        _ => vec![FeatureType::Hydrophobic],
    }
}
```

### 6.5 Pharmacophore Generation Pipeline

```rust
/// Generate pharmacophore model from spike events + receptor structure.
///
/// Algorithm:
///
/// 1. Load spike events for target site
/// 2. Cluster spikes by type using DBSCAN (eps=2.0Å, min_samples=5)
/// 3. For each cluster:
///    a. Compute centroid → feature position
///    b. Map spike type → feature type(s) using spike_type_to_features()
///    c. Compute weight from spike density (normalized 0-1)
///    d. For HBD/HBA features: compute direction vector from
///       centroid → nearest receptor N/O atom (points INTO pocket)
///    e. For Aromatic features: compute ring normal from PCA of spike positions
/// 4. Add receptor-derived features:
///    a. For each lining residue within 5Å of site centroid:
///       - ASP/GLU OD/OE → HBondAcceptor
///       - LYS NZ, ARG NH → HBondDonor
///       - SER/THR/TYR OH → HBondDonor + HBondAcceptor
///    b. These supplement spike-derived features
/// 5. Compute pairwise distance constraints between features
/// 6. Generate excluded volumes from receptor atoms within 3Å of any feature
/// 7. Merge overlapping features (same type, < 1.5Å apart)
/// 8. Output PharmacophoreModel
///
pub fn generate_pharmacophore(
    spike_events_path: &Path,
    binding_sites_path: &Path,
    receptor_pdb_path: &Path,
    site_id: u32,
    options: &PharmacophoreOptions,
) -> anyhow::Result<PharmacophoreModel>;

pub struct PharmacophoreOptions {
    pub cluster_eps: f64,          // DBSCAN epsilon for spike clustering. Default: 2.0
    pub cluster_min_samples: u32,  // DBSCAN min points. Default: 5
    pub feature_radius: f64,       // Tolerance sphere radius. Default: 1.5
    pub include_receptor: bool,    // Add receptor-derived features. Default: true
    pub include_exclusions: bool,  // Add excluded volumes. Default: true
    pub min_feature_weight: f64,   // Minimum weight to include. Default: 0.1
    pub max_features: u32,         // Cap on total features. Default: 20
    pub merge_distance: f64,       // Merge features closer than this. Default: 1.5
}
```

### 6.6 Pharmacophore Search — `search.rs`

```rust
/// Screen a compound database against a pharmacophore model.
///
/// For each molecule:
/// 1. Generate 3D conformer(s) via RDKit
/// 2. Perceive pharmacophore features via RDKit's
///    Chem.Pharm2D.Gobbi_Pharm2D or ChemicalFeatures
/// 3. Attempt rigid alignment to model features
/// 4. Score alignment: fraction of model features matched within tolerance
/// 5. Rank by alignment score
///
pub fn pharmacophore_screen(
    model: &PharmacophoreModel,
    database: &CompoundDatabase,
    options: &ScreenOptions,
) -> anyhow::Result<Vec<PharmacophoreHit>>;

pub struct CompoundDatabase {
    pub format: DatabaseFormat,
    pub path: PathBuf,
    pub count: Option<usize>,
}

pub enum DatabaseFormat {
    SmilesFile,       // One SMILES per line
    SdfFile,          // Multi-molecule SDF
    SdfGzFile,        // Gzipped SDF
    ZincSubset,       // ZINC format
}

pub struct ScreenOptions {
    pub min_match_fraction: f64,   // Fraction of features that must match. Default: 0.6
    pub max_rmsd: f64,             // Maximum RMSD of matched features. Default: 2.0
    pub num_conformers: u32,       // Conformers per molecule. Default: 50
    pub max_results: u32,          // Top N results. Default: 1000
    pub parallel_workers: u32,     // Python worker processes. Default: num_cpus
}

pub struct PharmacophoreHit {
    pub smiles: String,
    pub name: String,
    pub match_fraction: f64,       // 0-1, fraction of features matched
    pub alignment_rmsd: f64,       // RMSD of matched features (Å)
    pub matched_features: Vec<MatchedFeature>,
    pub unmatched_features: Vec<usize>,  // Indices of unmatched model features
}

pub struct MatchedFeature {
    pub model_index: usize,
    pub ligand_atoms: Vec<u32>,
    pub distance: f64,             // Distance between model and ligand feature (Å)
}
```

### 6.7 Output Format — `pharmacophore_model.json`

```json
{
  "name": "KRAS_G12C_site0_pharmacophore",
  "site_id": 0,
  "centroid": [-5.425, -22.45, 41.841],
  "features": [
    {
      "feature_type": "Hydrophobic",
      "position": [-3.2, -20.1, 40.5],
      "radius": 1.5,
      "direction": null,
      "weight": 0.95,
      "source": {
        "SpikeCluster": {
          "spike_type": "TRP",
          "spike_count": 87,
          "density_max": 142.3
        }
      }
    },
    {
      "feature_type": "HBondAcceptor",
      "position": [-6.8, -23.4, 42.1],
      "radius": 1.5,
      "direction": [0.32, -0.71, 0.62],
      "weight": 0.78,
      "source": {
        "ReceptorSurface": {
          "residue": "ASP-12-A",
          "atom_names": ["OD1", "OD2"]
        }
      }
    }
  ],
  "distance_constraints": [
    {
      "feature_i": 0,
      "feature_j": 1,
      "distance": 5.73,
      "tolerance": 1.0
    }
  ],
  "excluded_volumes": [
    {
      "position": [-4.1, -21.3, 41.0],
      "radius": 1.8
    }
  ],
  "metadata": {
    "receptor_pdb": "4obe_mono.pdb",
    "prism4d_version": "0.4.0",
    "generation_method": "spike_density",
    "spike_events_file": "4obe_mono.site0.spike_events.json",
    "num_spikes_total": 1523,
    "timestamp": "2026-02-11T15:30:00Z"
  }
}
```

### 6.8 Success Criteria — Phase 4

- [ ] Generates pharmacophore with 5-15 typed features per site
- [ ] Features include HBD, HBA, Hydrophobic, Aromatic (not just "density")
- [ ] Direction vectors computed for HBD/HBA features
- [ ] Known active compound matches pharmacophore (match_fraction > 0.6)
- [ ] Known inactive/decoy does NOT match (match_fraction < 0.3)
- [ ] Pharmacophore search against 10K SMILES completes in < 30 minutes
- [ ] Output JSON matches schema above
- [ ] PyMOL visualization shows features colored by type

---

## 7. Phase 5: Virtual Screening Pipeline

### 7.1 Objective

Multi-stage funnel that screens large compound libraries efficiently:
1. Pharmacophore pre-filter (fast, eliminates 90% of non-matches)
2. GPU docking of survivors (UniDock)
3. CNN rescoring of top poses (GNINA)
4. ADMET filtering (drug-likeness)
5. Final ranking and report

### 7.2 New Crate: `prism-screen`

```
crates/prism-screen/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── pipeline.rs          # Multi-stage screening pipeline
│   ├── funnel.rs            # Stage definitions and data flow
│   ├── admet.rs             # ADMET property filters
│   ├── diversity.rs         # Chemical diversity selection
│   ├── report.rs            # Screening report generation
│   └── bin/
│       └── prism_screen.rs  # CLI binary
├── python/
│   ├── admet_filter.py      # RDKit ADMET calculations
│   └── diversity_pick.py    # MaxMin diversity selection
└── tests/
    └── test_pipeline.rs
```

### 7.3 Pipeline Architecture

```rust
/// Virtual screening pipeline configuration.
pub struct ScreeningPipeline {
    pub stages: Vec<ScreeningStage>,
    pub input: CompoundDatabase,
    pub receptor: PreparedReceptor,
    pub binding_sites: Vec<DockingBox>,
    pub pharmacophore: Option<PharmacophoreModel>,
    pub output_dir: PathBuf,
}

pub enum ScreeningStage {
    /// Stage 1: Property filter (fast, CPU).
    PropertyFilter {
        max_mw: f64,              // Default: 500
        max_logp: f64,            // Default: 5.0
        max_hbd: u32,             // Default: 5
        max_hba: u32,             // Default: 10
        max_rotatable: u32,       // Default: 10
        max_tpsa: f64,            // Default: 140
        pains_filter: bool,       // Remove PAINS alerts. Default: true
    },

    /// Stage 2: Pharmacophore filter (medium speed, CPU).
    PharmacophoreFilter {
        model: PharmacophoreModel,
        min_match: f64,            // Default: 0.6
        max_rmsd: f64,             // Default: 2.0
        conformers: u32,           // Default: 50
    },

    /// Stage 3: GPU docking (slower, GPU).
    Docking {
        engine: DockingEngineType,
        exhaustiveness: u32,       // Default: 16 (lower for screening)
        top_n: u32,                // Keep top N per site. Default: 100
    },

    /// Stage 4: CNN rescoring (GPU).
    CnnRescore {
        model: GninaCnnModel,
        min_cnn_score: f64,        // Default: 0.5
    },

    /// Stage 5: Diversity selection.
    DiversityPick {
        num_diverse: u32,          // Default: 50
        fingerprint: FingerprintType,
    },
}

pub enum FingerprintType {
    Morgan2 { radius: u32 },      // Default radius 2 (ECFP4)
    RDKit,
    MACCS,
}

/// Run the full screening pipeline.
pub fn run_screening(
    pipeline: &ScreeningPipeline,
    progress: impl Fn(StageProgress),
) -> anyhow::Result<ScreeningResults>;

pub struct StageProgress {
    pub stage_name: String,
    pub stage_index: u32,
    pub total_stages: u32,
    pub compounds_in: u32,
    pub compounds_out: u32,
    pub elapsed_seconds: f64,
}

pub struct ScreeningResults {
    pub total_input: u32,
    pub stage_results: Vec<StageResult>,
    pub final_hits: Vec<ScreeningHit>,
    pub runtime_total: f64,
}

pub struct StageResult {
    pub stage_name: String,
    pub input_count: u32,
    pub output_count: u32,
    pub pass_rate: f64,
    pub runtime_seconds: f64,
}

pub struct ScreeningHit {
    pub rank: u32,
    pub smiles: String,
    pub name: String,
    pub consensus_score: f64,
    pub vina_affinity: Option<f64>,
    pub cnn_score: Option<f64>,
    pub cnn_affinity: Option<f64>,
    pub pharmacophore_match: Option<f64>,
    pub properties: MolecularProperties,
    pub pose_file: Option<PathBuf>,
}

pub struct MolecularProperties {
    pub mw: f64,
    pub logp: f64,
    pub hbd: u32,
    pub hba: u32,
    pub tpsa: f64,
    pub rotatable_bonds: u32,
    pub lipinski_violations: u32,
    pub pains_alerts: Vec<String>,
}
```

### 7.4 ADMET Filters — `admet.rs`

```rust
/// Drug-likeness filter rules.
///
/// Lipinski's Rule of Five:
///   MW ≤ 500, LogP ≤ 5, HBD ≤ 5, HBA ≤ 10
///   Allow 1 violation.
///
/// Veber's Rules (oral bioavailability):
///   Rotatable bonds ≤ 10, TPSA ≤ 140 Å²
///
/// PAINS (Pan-Assay Interference Compounds):
///   Filter out frequent hitters using RDKit's FilterCatalog.
///   ~480 substructure patterns.
///
/// Brenk Rules (unwanted substructures):
///   Reactive groups, toxic moieties, metabolic liabilities.
///
pub fn apply_admet_filter(
    molecules: &[PreparedLigand],
    rules: &AdmetRules,
) -> Vec<(usize, AdmetResult)>;

pub struct AdmetRules {
    pub lipinski: bool,            // Default: true
    pub lipinski_max_violations: u32, // Default: 1
    pub veber: bool,               // Default: true
    pub pains: bool,               // Default: true
    pub brenk: bool,               // Default: false
    pub custom_mw_range: Option<(f64, f64)>,
    pub custom_logp_range: Option<(f64, f64)>,
}

pub struct AdmetResult {
    pub passes: bool,
    pub lipinski_violations: u32,
    pub veber_violations: u32,
    pub pains_alerts: Vec<String>,
    pub brenk_alerts: Vec<String>,
    pub reason_failed: Option<String>,
}
```

### 7.5 Expected Throughput

```
Input: 1,000,000 compounds (ZINC subset)

Stage 1 — Property Filter:    ~100K/sec (CPU)     → ~200K survivors (20%)  [10 sec]
Stage 2 — Pharmacophore:      ~1K/sec (CPU, 8x)   → ~20K survivors (10%)   [200 sec]
Stage 3 — GPU Docking:        ~10K/hr (1 GPU)      → ~2K top poses          [2 hr]
Stage 4 — CNN Rescore:        ~50K/hr (1 GPU)      → ~500 high-confidence   [2 min]
Stage 5 — Diversity Pick:     instant               → 50 diverse hits        [1 sec]

Total: ~2.5 hours on single GPU for 1M compounds
```

### 7.6 Success Criteria — Phase 5

- [ ] 1M SMILES → 50 diverse hits in < 4 hours (single GPU)
- [ ] Funnel visualization shows clear attrition at each stage
- [ ] ADMET filter removes known PAINS compounds
- [ ] Top 50 hits are chemically diverse (Tanimoto < 0.7 pairwise)
- [ ] Report includes per-stage statistics and hit structures

---

## 8. Phase 6: Alchemical Free Energy (OpenFE)

### 8.1 Objective

Compute relative binding free energies (RBFE) for top screening hits to predict which compounds bind tightest. This is the gold standard for lead optimization.

### 8.2 OpenFE Overview

- **What**: Open-source alchemical free energy calculation toolkit
- **Method**: Thermodynamic cycle with lambda windows for alchemical transformations
- **Accuracy**: Approaches FEP+ (Schrödinger) on benchmarks
- **License**: MIT
- **Dependencies**: OpenMM (MIT/LGPL), perses, OpenFF
- **Repository**: https://github.com/OpenFreeEnergy/openfe

### 8.3 New Crate: `prism-fep`

```
crates/prism-fep/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── bridge.rs            # Rust ↔ Python/OpenFE bridge
│   ├── network.rs           # Perturbation network design
│   ├── analysis.rs          # FEP result analysis
│   ├── config.rs            # FEP configuration
│   └── bin/
│       └── prism_fep.rs     # CLI binary
├── python/
│   ├── fep_setup.py         # Network setup via OpenFE API
│   ├── fep_run.py           # Execute FEP calculations
│   ├── fep_analysis.py      # Analyze results, compute ΔΔG
│   └── fep_network.py       # LOMAP/RBFE network generation
└── tests/
    └── test_fep.rs
```

### 8.4 Data Structures

```rust
/// FEP calculation configuration.
pub struct FepConfig {
    pub receptor_pdb: PathBuf,
    pub ligands: Vec<FepLigand>,
    pub reference_ligand: String,     // Name of reference compound
    pub site_id: u32,
    pub binding_pose_source: PoseSource,
    pub simulation: FepSimulationConfig,
    pub output_dir: PathBuf,
}

pub struct FepLigand {
    pub name: String,
    pub smiles: String,
    pub docked_pose: Option<PathBuf>, // From Phase 2 docking
}

pub enum PoseSource {
    DockedPoses,           // Use poses from UniDock/GNINA
    CrystalStructure,      // Use co-crystal structure
    PharmacophoreAligned,  // Align to pharmacophore model
}

pub struct FepSimulationConfig {
    pub num_lambda_windows: u32,     // Default: 11
    pub simulation_length_ns: f64,   // Per window. Default: 5.0
    pub equilibration_ns: f64,       // Default: 1.0
    pub temperature_k: f64,          // Default: 300.0
    pub pressure_atm: f64,           // Default: 1.0
    pub force_field: String,         // Default: "openff-2.1.0"
    pub water_model: String,         // Default: "tip3p"
    pub gpu_device: u32,             // Default: 0
    pub num_replicas: u32,           // Default: 3
}

/// Perturbation network edge.
pub struct PerturbationEdge {
    pub ligand_a: String,
    pub ligand_b: String,
    pub similarity: f64,     // Tanimoto (ECFP4)
    pub n_heavy_atom_diff: u32,
    pub estimated_difficulty: FepDifficulty,
}

pub enum FepDifficulty {
    Easy,     // < 3 heavy atom change, high similarity
    Medium,   // 3-6 heavy atom change
    Hard,     // > 6 heavy atom change or ring changes
}

/// FEP result.
pub struct FepResult {
    pub edge: PerturbationEdge,
    pub ddg_kcal: f64,          // ΔΔG (A→B) in kcal/mol
    pub ddg_error: f64,         // Statistical uncertainty (kcal/mol)
    pub convergence: f64,       // 0-1, convergence metric
    pub overlap_matrix: Vec<Vec<f64>>,  // Lambda overlap
    pub runtime_hours: f64,
}

/// Full FEP analysis output.
pub struct FepAnalysis {
    pub reference_ligand: String,
    pub predictions: Vec<LigandPrediction>,
    pub cycle_closure_error: f64,   // Self-consistency check (kcal/mol)
    pub correlation_r2: Option<f64>, // R² vs experimental (if available)
}

pub struct LigandPrediction {
    pub name: String,
    pub smiles: String,
    pub dg_relative_kcal: f64,    // ΔG relative to reference
    pub dg_error: f64,
    pub predicted_rank: u32,
    pub experimental_ic50: Option<f64>,  // If available
}
```

### 8.5 OpenFE Python Bridge

```python
# python/fep_setup.py
#
# Input: JSON config with receptor, ligands, poses
# Output: OpenFE AlchemicalNetwork serialized to disk
#
# Steps:
# 1. Load receptor PDB → openfe.ProteinComponent
# 2. Load ligands (SDF with 3D poses) → openfe.SmallMoleculeComponent
# 3. Generate perturbation network via LOMAP (minimal spanning tree)
# 4. Create AlchemicalNetwork with RelativeHybridTopologyProtocol
# 5. Serialize network to disk for execution
#
# python/fep_run.py
#
# Input: Serialized AlchemicalNetwork
# Output: Per-edge ΔΔG results
#
# Steps:
# 1. Load network
# 2. For each edge: run RelativeHybridTopologyProtocol
#    - Build hybrid topology (ligand A → ligand B)
#    - Run lambda windows in parallel (OpenMM on GPU)
#    - Compute free energy via MBAR
# 3. Save raw results
#
# python/fep_analysis.py
#
# Input: Raw FEP results
# Output: Analyzed predictions with uncertainties
#
# Steps:
# 1. Construct thermodynamic graph
# 2. Compute MLE estimator for absolute ΔGs
# 3. Check cycle closure (should be < 1 kcal/mol)
# 4. Generate confidence intervals via bootstrap
# 5. Output ranked predictions
```

### 8.6 Success Criteria — Phase 6

- [ ] Set up perturbation network for 10 ligands automatically
- [ ] Run 1 FEP edge (A→B) on GPU in < 4 hours
- [ ] Full 10-ligand network in < 48 hours (parallelizable)
- [ ] ΔΔG predictions within 1 kcal/mol of experimental for validated set
- [ ] Cycle closure error < 1.5 kcal/mol
- [ ] Output predictions ranked by binding affinity

---

## 9. Phase 7: Unified CLI & Orchestration

### 9.1 Objective

Single `prism4d` command that orchestrates the entire pipeline from PDB → hits.

### 9.2 CLI Design

```
prism4d — GPU-accelerated drug discovery platform

SUBCOMMANDS:
  prep        Prepare receptor and ligands
  detect      Run binding site detection (NHS pipeline)
  dock        Dock ligands into detected sites
  rescore     Rescore poses with GNINA CNN
  pharma      Generate pharmacophore model
  screen      Run virtual screening pipeline
  fep         Run free energy perturbation
  report      Generate evidence pack

  pipeline    Run full pipeline (detect → dock → rescore → report)

USAGE:
  # Full pipeline
  prism4d pipeline -r protein.pdb --ligands library.sdf -o results/

  # Individual steps
  prism4d prep receptor -i protein.pdb -o protein.pdbqt
  prism4d prep ligand -s "CC(=O)Oc1ccccc1C(O)=O" -o aspirin.pdbqt
  prism4d detect -t protein.topology.json -o sites/
  prism4d dock -r protein.pdbqt --sites sites/binding_sites.json --ligands ligands/ -o docked/
  prism4d rescore -r protein.pdbqt --poses docked/ -o rescored/
  prism4d pharma -s sites/spike_events.json --sites sites/binding_sites.json -o pharma/
  prism4d screen -r protein.pdbqt --pharma pharma/model.json --library zinc.sdf -o screen/
  prism4d fep -r protein.pdb --ligands top_hits.sdf --poses docked/ -o fep/
  prism4d report -i results/ -o report/

GLOBAL OPTIONS:
  --gpu <ID>        CUDA device (default: 0)
  --threads <N>     CPU threads (default: auto)
  --verbose         Debug logging
  --quiet           Minimal output
  --json            JSON output to stdout
```

### 9.3 Configuration File — `prism4d.toml`

```toml
[general]
gpu_device = 0
threads = 0                    # 0 = auto-detect
log_level = "info"

[prep]
ph = 7.4
charge_method = "gasteiger"
add_hydrogens = true
remove_water = true

[detect]
steps = 500000
replicas = 3
multi_stream = 8

[dock]
engine = "auto"                # "unidock", "vina", "auto"
exhaustiveness = 32
num_modes = 20
box_padding = 5.0

[rescore]
cnn_model = "default"
min_cnn_score = 0.5

[pharma]
cluster_eps = 2.0
feature_radius = 1.5
max_features = 20

[screen]
max_mw = 500.0
max_logp = 5.0
pains_filter = true
diversity_pick = 50

[fep]
lambda_windows = 11
simulation_ns = 5.0
replicas = 3
```

---

## 10. Phase 8: Packaging & Distribution

### 10.1 Distribution Structure

```
prism4d-v0.4.0-linux-x86_64/
├── bin/
│   ├── prism4d              # Main CLI binary (Rust, static)
│   ├── nhs_rt_full          # Core detection binary
│   ├── pharmacophore_gpu    # GPU pharmacophore binary
│   ├── unidock              # UniDock binary (Apache 2.0)
│   ├── gnina                # GNINA binary (Apache 2.0)
│   └── vina                 # AutoDock Vina fallback (Apache 2.0)
├── lib/
│   ├── ptx/                 # Compiled CUDA kernels
│   │   ├── nhs_amber_fused.ptx
│   │   ├── pharmacophore_splat.ptx
│   │   └── ...
│   └── python/              # Python modules
│       ├── ligand_prep.py
│       ├── pharmacophore_perception.py
│       ├── admet_filter.py
│       ├── fep_setup.py
│       ├── fep_run.py
│       └── fep_analysis.py
├── data/
│   ├── amber/               # AMBER force field parameters
│   └── models/              # GNINA CNN model weights
├── config/
│   └── prism4d.toml         # Default configuration
├── THIRD_PARTY_LICENSES/
│   ├── UNIDOCK_LICENSE       # Apache 2.0
│   ├── GNINA_LICENSE         # Apache 2.0
│   ├── RDKIT_LICENSE         # BSD 3-Clause
│   ├── OPENFE_LICENSE        # MIT
│   ├── VINA_LICENSE          # Apache 2.0
│   └── OPENMM_LICENSE        # MIT + LGPL
├── LICENSE                   # PRISM4D proprietary license
├── README.md
└── VERSION
```

### 10.2 Python Environment

```bash
# Bundled conda environment or requirements.txt
# Required packages:
rdkit>=2024.03
meeko>=0.5
numpy>=1.24
scipy>=1.11
openfe>=1.0            # Phase 6 only
openff-toolkit>=0.15   # Phase 6 only
openmm>=8.1            # Phase 6 only
```

### 10.3 Build Script

```bash
#!/bin/bash
# build_release.sh

set -e

VERSION=$(cat VERSION)
TARGET="prism4d-v${VERSION}-linux-x86_64"

# 1. Build Rust binaries
cargo build --release -p prism-nhs --features prism-nhs/gpu
cargo build --release -p prism-dock
cargo build --release -p prism-pharma
cargo build --release -p prism-screen
cargo build --release -p prism-fep

# 2. Copy binaries
mkdir -p ${TARGET}/bin
cp target/release/prism4d ${TARGET}/bin/
cp target/release/nhs_rt_full ${TARGET}/bin/
cp target/release/pharmacophore_gpu ${TARGET}/bin/
cp target/release/prism_dock ${TARGET}/bin/
cp target/release/prism_screen ${TARGET}/bin/

# 3. Copy PTX kernels
mkdir -p ${TARGET}/lib/ptx
cp target/ptx/*.ptx ${TARGET}/lib/ptx/

# 4. Copy Python modules
mkdir -p ${TARGET}/lib/python
cp crates/prism-prep/python/*.py ${TARGET}/lib/python/
cp crates/prism-pharma/python/*.py ${TARGET}/lib/python/
cp crates/prism-screen/python/*.py ${TARGET}/lib/python/
cp crates/prism-fep/python/*.py ${TARGET}/lib/python/

# 5. Copy third-party binaries (pre-built)
cp vendor/unidock ${TARGET}/bin/
cp vendor/gnina ${TARGET}/bin/
cp vendor/vina ${TARGET}/bin/

# 6. Copy licenses
mkdir -p ${TARGET}/THIRD_PARTY_LICENSES
# ... copy license files

# 7. Package
tar czf ${TARGET}.tar.gz ${TARGET}/
sha256sum ${TARGET}.tar.gz > ${TARGET}.tar.gz.sha256
```

### 10.4 System Requirements

```
Minimum:
  - NVIDIA GPU: Compute Capability 7.0+ (Volta/Turing/Ampere/Ada/Blackwell)
  - VRAM: 8 GB
  - CUDA: 12.0+
  - RAM: 16 GB
  - Storage: 50 GB (with compound libraries)
  - Python: 3.10+
  - OS: Linux x86_64 (Ubuntu 22.04+ / RHEL 8+)

Recommended:
  - NVIDIA GPU: RTX 4090 / A100 / H100
  - VRAM: 24 GB+
  - RAM: 64 GB+
  - Storage: 500 GB SSD
  - CPU: 16+ cores
```

---

## 11. Data Flow Specification

### 11.1 File Format Chain

```
PDB ──→ Topology JSON ──→ binding_sites.json ──→ spike_events.json
 │                                │                      │
 ▼                                ▼                      ▼
PDBQT (receptor)          DockingBox configs      PharmacophoreModel
 │                                │                      │
 │    ┌───────────────────────────┘                      │
 │    │                                                  │
 ▼    ▼                                                  ▼
UniDock docking ──→ docked_poses.pdbqt          pharma_screen hits
       │                    │                          │
       ▼                    ▼                          │
  GNINA rescore ──→ consensus_ranked.json              │
       │                                               │
       └────────────────────┬──────────────────────────┘
                            ▼
                    top_hits.json
                            │
                            ▼
                    OpenFE FEP ──→ ddg_predictions.json
                            │
                            ▼
                    final_report.html
```

### 11.2 Inter-Phase Data Contracts

```
Phase 1 → Phase 2:
  receptor.pdbqt          (PDBQT format, AD4 atom types)
  ligands/*.pdbqt         (PDBQT format, 3D, Gasteiger charges)
  docking_boxes.json      (center, size, exhaustiveness per site)

Phase 2 → Phase 3:
  docked/*_site*_docked.pdbqt  (multi-MODEL PDBQT)
  docking_results.json         (scores, per-site per-ligand)

Phase 3 → Phase 4:
  rescored_results.json        (Vina + CNN consensus scores)
  top_poses/*.pdbqt            (best poses for pharmacophore)

Phase 4 → Phase 5:
  pharmacophore_model.json     (typed features + constraints)
  density_grids/*.dx           (GPU splatted density maps)

Phase 5 → Phase 6:
  screening_hits.json          (top 50-100 diverse hits)
  hit_poses/*.pdbqt            (docked poses for FEP)

Phase 6 → Report:
  fep_predictions.json         (ΔΔG per compound)
  fep_network.json             (perturbation graph)
```

---

## 12. File Manifest

### New Files to Create

| Phase | File | Type | Lines (est.) |
|-------|------|------|-------------|
| 1 | `crates/prism-prep/Cargo.toml` | Config | 40 |
| 1 | `crates/prism-prep/src/lib.rs` | Rust | 30 |
| 1 | `crates/prism-prep/src/receptor.rs` | Rust | 400 |
| 1 | `crates/prism-prep/src/ligand.rs` | Rust | 200 |
| 1 | `crates/prism-prep/src/pdbqt.rs` | Rust | 300 |
| 1 | `crates/prism-prep/src/atom_typing.rs` | Rust | 350 |
| 1 | `crates/prism-prep/src/charges.rs` | Rust | 150 |
| 1 | `crates/prism-prep/src/protonation.rs` | Rust | 200 |
| 1 | `crates/prism-prep/src/bin/prism_prep.rs` | Rust | 100 |
| 1 | `crates/prism-prep/python/ligand_prep.py` | Python | 200 |
| 2 | `crates/prism-dock/Cargo.toml` | Config | 35 |
| 2 | `crates/prism-dock/src/lib.rs` | Rust | 20 |
| 2 | `crates/prism-dock/src/engine.rs` | Rust | 80 |
| 2 | `crates/prism-dock/src/unidock.rs` | Rust | 250 |
| 2 | `crates/prism-dock/src/vina.rs` | Rust | 200 |
| 2 | `crates/prism-dock/src/config.rs` | Rust | 100 |
| 2 | `crates/prism-dock/src/results.rs` | Rust | 200 |
| 2 | `crates/prism-dock/src/pose.rs` | Rust | 100 |
| 2 | `crates/prism-dock/src/batch.rs` | Rust | 150 |
| 2 | `crates/prism-dock/src/bin/prism_dock.rs` | Rust | 120 |
| 3 | `crates/prism-dock/src/gnina.rs` | Rust | 200 |
| 3 | `crates/prism-dock/src/consensus.rs` | Rust | 150 |
| 4 | `crates/prism-pharma/Cargo.toml` | Config | 35 |
| 4 | `crates/prism-pharma/src/lib.rs` | Rust | 20 |
| 4 | `crates/prism-pharma/src/model.rs` | Rust | 200 |
| 4 | `crates/prism-pharma/src/perception.rs` | Rust | 300 |
| 4 | `crates/prism-pharma/src/spike_features.rs` | Rust | 250 |
| 4 | `crates/prism-pharma/src/alignment.rs` | Rust | 200 |
| 4 | `crates/prism-pharma/src/search.rs` | Rust | 250 |
| 4 | `crates/prism-pharma/src/io.rs` | Rust | 150 |
| 4 | `crates/prism-pharma/src/bin/prism_pharma.rs` | Rust | 100 |
| 4 | `crates/prism-pharma/python/pharmacophore_perception.py` | Python | 300 |
| 4 | `crates/prism-pharma/python/pharmacophore_search.py` | Python | 250 |
| 5 | `crates/prism-screen/Cargo.toml` | Config | 35 |
| 5 | `crates/prism-screen/src/lib.rs` | Rust | 20 |
| 5 | `crates/prism-screen/src/pipeline.rs` | Rust | 400 |
| 5 | `crates/prism-screen/src/funnel.rs` | Rust | 200 |
| 5 | `crates/prism-screen/src/admet.rs` | Rust | 150 |
| 5 | `crates/prism-screen/src/diversity.rs` | Rust | 100 |
| 5 | `crates/prism-screen/src/report.rs` | Rust | 200 |
| 5 | `crates/prism-screen/src/bin/prism_screen.rs` | Rust | 120 |
| 5 | `crates/prism-screen/python/admet_filter.py` | Python | 200 |
| 5 | `crates/prism-screen/python/diversity_pick.py` | Python | 100 |
| 6 | `crates/prism-fep/Cargo.toml` | Config | 30 |
| 6 | `crates/prism-fep/src/lib.rs` | Rust | 20 |
| 6 | `crates/prism-fep/src/bridge.rs` | Rust | 200 |
| 6 | `crates/prism-fep/src/network.rs` | Rust | 150 |
| 6 | `crates/prism-fep/src/analysis.rs` | Rust | 200 |
| 6 | `crates/prism-fep/src/config.rs` | Rust | 100 |
| 6 | `crates/prism-fep/src/bin/prism_fep.rs` | Rust | 100 |
| 6 | `crates/prism-fep/python/fep_setup.py` | Python | 300 |
| 6 | `crates/prism-fep/python/fep_run.py` | Python | 250 |
| 6 | `crates/prism-fep/python/fep_analysis.py` | Python | 200 |
| 6 | `crates/prism-fep/python/fep_network.py` | Python | 150 |
| 7 | `crates/prism-cli/src/bin/prism4d_main.rs` | Rust | 300 |
| 8 | `build_release.sh` | Shell | 80 |

**Total new files**: ~55
**Total new lines**: ~8,500 Rust + ~2,000 Python ≈ **10,500 lines**

### Files to Modify

| File | Change |
|------|--------|
| `Cargo.toml` (workspace) | Add 5 new crate members |
| `crates/prism-nhs/src/lib.rs` | No change needed |
| `crates/prism-report/src/finalize.rs` | Add docking results to evidence pack |

---

## 13. Dependency Matrix

### Rust Dependencies (New)

| Crate | prism-prep | prism-dock | prism-pharma | prism-screen | prism-fep |
|-------|-----------|-----------|-------------|-------------|----------|
| anyhow | x | x | x | x | x |
| serde + serde_json | x | x | x | x | x |
| clap | x | x | x | x | x |
| log + env_logger | x | x | x | x | x |
| rayon | x | x | x | x | — |
| tempfile | x | x | — | — | — |
| regex | x | — | — | — | — |
| prism-io | x | x | x | x | x |
| prism-dock | — | — | — | x | x |
| prism-pharma | — | — | — | x | — |
| prism-prep | — | x | x | x | x |

### External Binaries

| Binary | Source | License | Phase | Required? |
|--------|--------|---------|-------|-----------|
| `unidock` | github.com/dptech-corp/Uni-Dock | Apache 2.0 | 2 | Yes (or vina) |
| `gnina` | github.com/gnina/gnina | Apache 2.0 | 3 | Recommended |
| `vina` | vina.scripps.edu | Apache 2.0 | 2 | Fallback |

### Python Packages

| Package | Version | License | Phase | Required? |
|---------|---------|---------|-------|-----------|
| rdkit | ≥2024.03 | BSD 3-Clause | 1,4,5 | Yes |
| meeko | ≥0.5 | Apache 2.0 | 1 | Yes |
| numpy | ≥1.24 | BSD | All | Yes |
| scipy | ≥1.11 | BSD | 4,6 | Yes |
| openfe | ≥1.0 | MIT | 6 | Phase 6 only |
| openff-toolkit | ≥0.15 | MIT | 6 | Phase 6 only |
| openmm | ≥8.1 | MIT+LGPL | 6 | Phase 6 only |

---

## 14. Testing Strategy

### 14.1 Unit Tests (Per Crate)

| Test | What it validates |
|------|-------------------|
| `test_ad4_typing_ala` | ALA residue: all C atoms → `C`, backbone N → `N`, backbone O → `OA` |
| `test_ad4_typing_phe` | PHE ring carbons → `A`, non-ring → `C` |
| `test_ad4_typing_his` | HIS ND1/NE2 protonation-dependent typing |
| `test_ad4_typing_cys_free` | CYS SG free → `SA` |
| `test_ad4_typing_cys_disulfide` | CYS SG in SS bond → `S` |
| `test_ligand_prep_aspirin` | SMILES → 3D PDBQT with correct atom count |
| `test_pdbqt_round_trip` | Write PDBQT → read back → same coordinates ±0.001 |
| `test_docking_box_from_sites` | binding_sites.json → correct box dimensions |
| `test_vina_config_format` | Config file matches Vina expected format exactly |
| `test_parse_docked_pdbqt` | Parse multi-MODEL PDBQT → correct scores and poses |
| `test_gnina_score_parse` | Parse GNINA output → CNN scores extracted |
| `test_consensus_ranking` | Consensus reranks when CNN disagrees with Vina |
| `test_spike_to_features` | Each spike type maps to correct pharmacophore features |
| `test_pharmacophore_json` | Serialization round-trip for pharmacophore model |
| `test_lipinski_filter` | Known drug passes, known non-drug fails |
| `test_pains_detection` | Known PAINS compound flagged |

### 14.2 Integration Tests

| Test | Input | Expected Output |
|------|-------|-----------------|
| `e2e_prep_and_dock` | 1btl.pdb + aspirin SMILES | Negative affinity score |
| `e2e_kras_g12c_sotorasib` | KRAS G12C PDB + sotorasib | Affinity < -7 kcal/mol |
| `e2e_full_pipeline` | 4obe.pdb + 100 SMILES | Ranked hits with report |
| `e2e_pharmacophore_search` | Model + ZINC subset | Actives ranked above decoys |
| `e2e_fep_benzene_phenol` | Simple pair | ΔΔG within 1 kcal/mol of experiment |

### 14.3 Validation Targets

| Target | PDB | Known Drug | Expected Affinity |
|--------|-----|-----------|-------------------|
| KRAS G12C | 6OIM | Sotorasib | -9 to -11 kcal/mol |
| EGFR | 1M17 | Erlotinib | -8 to -10 kcal/mol |
| CDK2 | 1H1Q | Roscovitine | -7 to -9 kcal/mol |
| HIV protease | 1HVR | Indinavir | -10 to -12 kcal/mol |
| COX-2 | 1CX2 | Celecoxib | -8 to -10 kcal/mol |

---

## 15. Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| UniDock binary incompatible with our CUDA version | Phase 2 blocked | Build from source with matching CUDA; keep Vina as fallback |
| GNINA model weights too large for distribution | Package size | Download on first run, cache locally |
| RDKit Python ↔ Rust subprocess overhead | Screening speed | Batch ligands (100 per subprocess call), share conformer cache |
| OpenFE requires specific OpenMM version | Phase 6 conflicts | Isolate in conda env, call via subprocess |
| PTX compiled for sm_120 doesn't run on older GPUs | User compatibility | Ship multiple PTX targets (sm_75, sm_86, sm_89, sm_120) |
| Pharmacophore search too slow for 1M+ compounds | Screening time | Pre-filter by MW/LogP, use RDKit fingerprint pre-screen |
| False positives from spike density maps | Bad pharmacophore | Validate against known actives/inactives before release |
| Licensing dispute over modified UniDock code | Legal | Keep modifications in separate wrapper layer, don't modify UniDock source |

---

## Implementation Order

```
Phase 1 (Week 1-2):  Receptor + Ligand Preparation
Phase 2 (Week 2-3):  UniDock GPU Docking Integration
Phase 3 (Week 3-4):  GNINA CNN Rescoring
Phase 4 (Week 4-6):  Pharmacophore Model Generation
Phase 5 (Week 6-8):  Virtual Screening Pipeline
Phase 6 (Week 8-12): Alchemical Free Energy
Phase 7 (Week 10-12): Unified CLI
Phase 8 (Week 12-14): Packaging & Distribution
```

Phases 1-3 are sequential (each depends on previous).
Phases 4 and 2-3 can partially overlap.
Phase 5 requires 2, 3, and 4.
Phase 6 requires 2 and 5.
Phases 7-8 are integration/polish.

---

*End of Blueprint*
