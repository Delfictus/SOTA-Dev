# PRISM-4D Explicit Solvent Implementation Plan (Hybrid, Failure-Resistant)

**Branch:** `feature/explicit-solvent`
**Working Directory:** `~/Desktop/PRISM4D-dev`
**Stable Reference (DO NOT MODIFY):** `~/Desktop/PRISM4D-v1.1.0-STABLE`

---
## SOVEREIGNTY DECLARATION

PRISM-4D explicit solvent requires **ZERO** external dependencies:
- ❌ No AmberTools/tleap
- ❌ No GROMACS
- ❌ No OpenMM
- ❌ No packmol

All solvation, parameterization, and simulation handled internally.

## SECTION 1: DEFAULTS VS LOCKED PARAMETERS

### A) Shared Defaults (Both Implicit & Explicit)

| Parameter | Default Value | User-Overridable | Location |
|-----------|--------------|------------------|----------|
| Timestep | 2.0 fs | Yes (`--dt`) | CLI |
| Temperature | 310 K | Yes (`--temperature`) | CLI |
| Non-bonded cutoff | 12 Å | No (engine) | `amber_mega_fused.cu:106` |
| Langevin γ | 0.01 fs⁻¹ | Yes (`--gamma`) | CLI |
| Integrator | Velocity Verlet | No (engine) | `run_verlet()` |
| k_B | 0.001987204 kcal/(mol·K) | No (physics) | CUDA kernel |
| Cell size | 10 Å | No (engine) | `CELL_SIZE` |
| Neighbor rebuild | 50 steps | No (engine) | `NEIGHBOR_REBUILD_INTERVAL` |

### B) Implicit-Solvent Defaults

| Parameter | Default | Notes |
|-----------|---------|-------|
| Dielectric | ε = 4r | Distance-dependent, `IMPLICIT_SOLVENT_SCALE = 0.25` |
| Position restraints | ON (k=1.0) | Prevents unfolding without water cage |
| PBC | OFF | Non-periodic (no periodic images) |
| PME | OFF | Uses screened Coulomb |
| COM removal | OFF | N/A for non-periodic |

### C) Explicit-Solvent Defaults

| Parameter | Default | Notes |
|-----------|---------|-------|
| Dielectric | ε = 1 | Explicit water + PME; no distance-dependent screening |
| Position restraints | OFF | See note below |
| PBC | ON | Orthorhombic box required |
| PME | ON | Long-range electrostatics |
| COM removal | ON | Prevents drift in periodic system |
| SETTLE | ON | Rigid TIP3P water |
| H-constraints | ON | SHAKE-like for protein X-H bonds |

**Restraints Note:** Explicit solvent provides a realistic aqueous environment; restraints default OFF for production runs, but optional gentle restraints (k ~ 0.1–1.0 kcal/(mol·Å²)) are allowed during early equilibration to prevent initial unfolding artifacts. Restraints are NOT required for stability once equilibrated.

### D) DOF Calculation

**Current implementation:**
```
N_dof = 3 * N_atoms (implicit, no constraints)
T = 2 * KE / (N_dof * k_B)
```

**For explicit solvent (Phase 0):**
```
N_dof = 3 * N_atoms - 3 (COM removal)
      - 3 * N_waters (SETTLE removes 3 DOF per water)
      - N_h_constraints (each X-H constraint removes 1 DOF)
```

**Limitation:** Current code uses `3 * N_atoms`. Phase 0 will add proper DOF counting.

---

## SECTION 2: CURRENT ARCHITECTURE SUMMARY

### GPU Memory Layout

**Structure of Arrays (SoA):**
- `d_positions`: `[n_atoms * 3]` flat f32 (x0, y0, z0, x1, y1, z1, ...)
- `d_velocities`: `[n_atoms * 3]` flat f32
- `d_forces`: `[n_atoms * 3]` flat f32
- `d_nb_sigma`, `d_nb_epsilon`, `d_nb_charge`, `d_nb_mass`: `[n_atoms]` separate arrays

**Bonded terms:** Flat packed arrays (`d_bond_atoms`, `d_bond_params`, etc.)

**Cell/Neighbor lists:**
- `d_cell_list`: `[MAX_TOTAL_CELLS * MAX_ATOMS_PER_CELL]` spatial bins
- `d_neighbor_list`: `[n_atoms * NEIGHBOR_LIST_SIZE]` per-atom neighbors
- `d_n_neighbors`: `[n_atoms]` neighbor counts

### Kernel Responsibilities

| Kernel | Purpose |
|--------|---------|
| `compute_forces_only` | Bonded + short-range NB forces |
| `velocity_verlet_step1` | v += (dt/2)*a; x += dt*v |
| `velocity_verlet_step2` | v += (dt/2)*a; Langevin thermostat |
| `settle_constraints` | SETTLE for rigid water |
| `build_cell_list` | Spatial binning |
| `build_neighbor_list` | Per-atom neighbor generation |
| `set_pbc_box` | Set box dimensions on GPU |
| `apply_position_restraints` | Harmonic restraints |

### Integration Step Ordering (`run_verlet`)

```
for step in 0..n_steps:
    1. settle.save_positions()          // Store for SETTLE projection
    2. memset_zeros(forces)             // Clear force buffer
    3. apply_position_restraints()      // If enabled
    4. compute_forces_only()            // Bonded + short-range NB
    5. vv_step1()                       // Half-kick + drift
    6. memset_zeros(forces)             // Clear for 2nd eval
    7. apply_position_restraints()      // At new position
    8. compute_forces_only()            // 2nd force evaluation
    9. vv_step2()                       // Half-kick + thermostat
   10. settle.apply()                   // Constrain waters
   11. h_constraints.apply()            // Constrain X-H bonds
   12. sample_energy()                  // Every 10 steps
   13. log()                            // Every 100 steps
```

### Current Cell-List/Neighbor-List Status

**Already implemented:**
- Cell size: 10 Å (line 139 amber_mega_fused.cu)
- `build_cell_list` kernel: assigns atoms to spatial bins
- `build_neighbor_list` kernel: generates per-atom neighbor lists
- Rebuild interval: every 50 steps
- Cutoff: 12 Å with neighbor search at 10 Å cell size

**Gap:** No skin distance tracking, no displacement-based rebuild trigger.

### PBC Status

**Already implemented in CUDA:**
```cuda
// NOTE: Should be __constant__ for read-only parameters (perf + safety)
// Current code may use __device__; Phase 1 should migrate to:
__constant__ float3 d_box_dims;
__constant__ float3 d_box_inv;
__constant__ int d_use_pbc;

__device__ float3 apply_pbc(float3 dr) {
    if (d_use_pbc) {
        dr.x -= d_box_dims.x * rintf(dr.x * d_box_inv.x);
        // ...
    }
    return dr;
}

__device__ float3 wrap_position(float3 pos) { ... }
```

**Gaps:**
1. `wrap_position()` exists but is not called after integration
2. Box constants may use `__device__` instead of `__constant__` memory (Phase 1 fix)

### PME Status

**Already implemented:**
- `crates/prism-gpu/src/pme.rs` (400+ lines)
- `crates/prism-gpu/src/kernels/pme.cu` (300+ lines)
- B-spline order: 4
- Grid spacing: 1.0 Å
- Ewald β: **MUST BE COMPUTED** (see note below)

**Ewald Beta Computation (MANDATORY - CANONICAL DEFINITION):**
```rust
/// Compute Ewald splitting parameter from tolerance and cutoff
/// β = sqrt(-ln(tolerance)) / cutoff
///
/// Example: tolerance=1e-5, cutoff=12 Å
///   β = sqrt(-ln(1e-5)) / 12 = sqrt(11.51) / 12 ≈ 0.283 Å⁻¹
fn compute_ewald_beta(cutoff: f32, tolerance: f32) -> f32 {
    (-tolerance.ln()).sqrt() / cutoff
}
```

**Action Required:**
- Remove any hardcoded β values (e.g., 0.34) from code and documentation
- Always compute β at runtime using this formula
- Store tolerance as the user-facing parameter, not β

**Gaps:**
1. No standalone validation harness (PME integrated but not independently tested)
2. β may be hardcoded instead of computed from tolerance/cutoff

### Explicit Insertion Points

| Component | Insertion Point | Status |
|-----------|-----------------|--------|
| PBC minimum image | `apply_pbc()` in force kernels | ✅ Implemented |
| Position wrapping | **After constraints** (SETTLE + H-constraints) | ❌ Not called |
| COM drift removal | After integration, before save | ❌ Not implemented |
| Neighbor lists | Before force computation | ✅ Implemented |
| PME forces | Before `compute_forces_only()` | ✅ Implemented |

---

## SECTION 3: PHASE 0 — DIAGNOSTICS LOGGING (IMMEDIATE)

**Purpose:** Publication figures + debugging for explicit solvent.

### Deliverables

1. `energy_timeseries.csv`:
   ```csv
   step,time_ps,potential_kcal,kinetic_kcal,total_kcal
   0,0.000,-15234.5,1523.2,-13711.3
   100,0.200,-15241.2,1518.9,-13722.3
   ```

2. `temperature_timeseries.csv`:
   ```csv
   step,time_ps,temperature_K,n_dof,n_atoms,n_waters,n_settle_constraints,n_h_constraints
   0,0.000,310.5,9234,3500,1000,3000,234
   100,0.200,309.6,9234,3500,1000,3000,234
   ```

   **Rationale:** Including constraint counts makes "temperature off by 10–30 K" bugs reproducible from the CSV alone without needing stdout logs.

### Temperature Formula

```
T = (2 * E_kin) / (N_dof * k_B)
k_B = 0.001987204 kcal/(mol·K)

N_dof = 3 * N_atoms - 3              // COM removal
      - 3 * N_waters                  // SETTLE (3 holonomic constraints per water)
      - N_h_constraints               // X-H bonds (1 constraint per bond)
```

**Constraint Counting Verification (REQUIRED):**
- SETTLE: 3 constraints per water molecule (O-H1 distance, O-H2 distance, H1-H2 distance)
- H-constraints: 1 constraint per X-H bond (fixes bond length)

Add explicit logging to verify constraint counts:
```rust
let n_settle_constraints = 3 * n_waters;
let n_h_constraints = h_constraints.map_or(0, |h| h.n_constraints());
info!("DOF accounting: {} atoms, {} waters ({} SETTLE constraints), {} H-constraints",
      n_atoms, n_waters, n_settle_constraints, n_h_constraints);
info!("N_dof = {} - 3 - {} - {} = {}",
      3 * n_atoms, n_settle_constraints, n_h_constraints, n_dof);

// Sanity check: verify constraint counts match topology
assert_eq!(n_settle_constraints, 3 * topology.n_waters());
assert_eq!(n_h_constraints, h_clusters.iter().map(|c| c.len()).sum());
```

This prevents subtle "temperature off by 10-30 K" bugs from incorrect DOF counting.

### Files to Modify

| File | Changes | Est. LOC |
|------|---------|----------|
| `amber_mega_fused.rs` | Add `EnergyRecord` struct, DOF calculation | +80 |
| `generate_ensemble.rs` | Add `--energy-log`, `--temperature-log` CLI args | +60 |
| `generate_ensemble.rs` | Write CSV after simulation | +40 |

### Implementation Details

```rust
// amber_mega_fused.rs
pub struct EnergyRecord {
    pub step: u64,
    pub time_ps: f64,
    pub potential_energy: f64,
    pub kinetic_energy: f64,
    pub temperature: f64,
}

pub struct HmcRunResult {
    // ... existing fields ...
    pub energy_trajectory: Vec<EnergyRecord>,
    pub n_dof: usize,  // Degrees of freedom
}

fn compute_n_dof(&self) -> usize {
    let base = 3 * self.n_atoms;
    let com_removal = 3;  // Explicit only
    let settle_dof = 3 * self.settle.map_or(0, |s| s.n_waters());
    let h_constraint_dof = self.h_constraints.map_or(0, |h| h.n_constraints());
    base - com_removal - settle_dof - h_constraint_dof
}
```

### Acceptance Criteria

- [ ] Implicit-solvent runs produce identical trajectories (regression test)
- [ ] Energy CSV written with correct format
- [ ] Temperature matches expected ~310K for equilibrated system
- [ ] DOF count matches: 3*N - 3 - 3*waters - h_constraints

### Commit

```
feat(simulation): Add energy and temperature logging with DOF accounting
```

---

## SECTION 3B: PHASE 0.5 — IMPLICIT SOLVENT REGRESSION TEST

**Purpose:** Establish a reference trajectory before touching any integration code. This ensures implicit solvent mode remains unchanged throughout explicit solvent development.

### Setup (One-Time)

```bash
# Save current implicit solvent trajectory as reference
cargo run --release -p prism-validation --bin generate-ensemble -- \
    --topology test_data/6m0j.json \
    --steps 10000 \
    --temperature 310 \
    --output reference_implicit.pdb \
    --energy-log reference_implicit_energy.csv

# Store checksums
md5sum reference_implicit.pdb > reference_implicit.md5
md5sum reference_implicit_energy.csv >> reference_implicit.md5
```

### After Each Phase

```bash
# Re-run implicit solvent with same seed
cargo run --release -p prism-validation --bin generate-ensemble -- \
    --topology test_data/6m0j.json \
    --steps 10000 \
    --temperature 310 \
    --output new_implicit.pdb \
    --energy-log new_implicit_energy.csv

# Verify trajectories match (should be bit-identical if RNG seeded)
diff reference_implicit.pdb new_implicit.pdb
diff reference_implicit_energy.csv new_implicit_energy.csv
```

### Acceptance Criteria

- [ ] Reference trajectory saved before any Phase 1+ changes
- [ ] Implicit mode produces bit-identical results after each phase
- [ ] If not bit-identical, verify energy/temperature within tolerance (< 0.01%)

### Note

If RNG seeding is not deterministic, compare statistically:
- Mean temperature within 0.1 K
- Mean energy within 0.1 kcal/mol
- RMSD trajectory envelope similar

---

## SECTION 4: PHASE 1 — PBC CORE

### Components to Implement

1. **Minimum image convention** — ✅ Already in `apply_pbc()`
2. **Position wrapping** — Need to call `wrap_position()` after constraints
3. **COM drift removal** — New kernel needed
4. **Device-side box constants** — ⚠️ Migrate from `__device__` to `__constant__` (performance + safety)

### Position Wrapping Integration

**Ordering Decision:** Wrap positions AFTER constraints (SETTLE + H-constraints), not before.

**Rationale:**
- SETTLE computes constrained positions relative to old positions
- If we wrap before constraints, atoms at box boundary may have discontinuous old→new displacements
- Wrapping after constraints keeps constrained geometry intact, then places molecule inside box
- Minimum image convention in force kernels uses wrapped coordinates consistently

```rust
// In run_verlet(), AFTER SETTLE and H-constraints:
if self.pbc_enabled {
    self.wrap_positions()?;  // Wrap after constraints
}
```

**Updated Integration Ordering:**
```
9. vv_step2()                       // Half-kick + thermostat
10. settle.apply()                   // Constrain waters
11. h_constraints.apply()            // Constrain X-H bonds
12. wrap_positions()                 // NEW: wrap after constraints
13. sample_energy()                  // Every 10 steps
```

New Rust method:
```rust
pub fn wrap_positions(&mut self) -> Result<()> {
    // Launch wrap_positions kernel
}
```

### COM Drift Removal

**Frequency:** Every 10–100 steps is sufficient and faster (default: every 10 steps). Every-step removal is unnecessary and adds overhead.

**Ordering:** After `vv_step2` + constraints + wrap:
```
vv_step2()           // Half-kick + thermostat
settle.apply()       // Constrain waters
h_constraints.apply() // Constrain X-H bonds
wrap_positions()     // Wrap into box
remove_com_drift()   // Remove net momentum from velocities
```

**Rationale:** Applying COM removal after constraints prevents constraints from reintroducing net drift via numerical noise.

**Important:** COM removal applies to velocities only. Positions are wrapped; do not recenter positions unless debugging.

New CUDA kernel:
```cuda
__global__ void remove_com_drift(
    float* velocities,  // Velocities only, not positions
    const float* masses,
    int n_atoms
) {
    // 1. Compute COM velocity (parallel reduction)
    // 2. Subtract from all velocities
}
```

### Files to Modify

| File | Changes | Est. LOC |
|------|---------|----------|
| `amber_mega_fused.rs` | Add `wrap_positions()` method | +30 |
| `amber_mega_fused.cu` | Add `wrap_positions_kernel` | +20 |
| `amber_mega_fused.cu` | Add `remove_com_drift` kernel | +50 |
| `amber_mega_fused.cu` | Migrate box vars to `__constant__` | +10 |
| `amber_mega_fused.rs` | Call in `run_verlet()` | +10 |

### Acceptance Criteria

- [ ] Single particle wraps correctly at box boundaries
- [ ] Implicit mode completely unaffected (bypass when `!pbc_enabled`)
- [ ] Atom coordinates stay within [0, L) after wrapping
- [ ] COM velocity = 0 after removal (for explicit solvent)
- [ ] Box constants use `__constant__` memory (not `__device__`)

### Commit

```
feat(gpu): Add PBC position wrapping and COM drift removal
```

---

## SECTION 5: PHASE 2 — NEIGHBOR LISTS (REQUIRED BEFORE PME)

**Critical:** This phase must complete before PME integration testing.

### Current State

Neighbor lists exist but need verification for PBC correctness.

### Verlet List Enhancement

**Current:** Fixed 50-step rebuild interval.

**Target:** Displacement-based rebuild with PBC-aware per-atom tracking:
```rust
// Store positions at last neighbor list build (GPU buffer)
d_pos_at_build: CudaSlice<f32>,  // [n_atoms * 3]

// On rebuild:
stream.memcpy_dtod(&d_positions, &mut d_pos_at_build)?;

// Each step, compute max displacement with minimum image:
let max_displacement = compute_max_displacement_pbc();  // GPU kernel
if max_displacement > skin_distance / 2.0 {
    rebuild_neighbor_lists();
}
```

**CUDA kernel for PBC-aware displacement:**
```cuda
__global__ void compute_max_displacement(
    const float* pos,           // Current positions
    const float* pos_at_build,  // Positions when neighbor list was built
    float* max_disp,            // Output: max displacement (atomic reduce)
    int n_atoms
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_atoms) return;

    float3 dr = apply_pbc(make_float3(
        pos[i*3]   - pos_at_build[i*3],
        pos[i*3+1] - pos_at_build[i*3+1],
        pos[i*3+2] - pos_at_build[i*3+2]
    ));
    float disp = sqrtf(dot3(dr, dr));
    atomicMax((int*)max_disp, __float_as_int(disp));  // Float atomic max trick
}
```

**Why PBC-aware:** Without minimum image, boundary crossings cause spurious large displacements (e.g., atom wraps from x=0.1 to x=L-0.1 shows |Δx|≈L instead of 0.2).

### Parameters

| Parameter | Value | Derivation |
|-----------|-------|------------|
| Cutoff (r_c) | 12 Å | AMBER standard |
| Skin distance | 2.5 Å | Default, user-overridable (consider 2.0 Å for denser systems) |
| Neighbor radius | 14.5 Å | r_c + skin |
| Cell size | 10 Å | Existing, adequate |

### PBC-Aware Neighbor Search

The `build_neighbor_list` kernel must use minimum image convention and build to `cutoff + skin`:
```cuda
// NEIGHBOR LIST BUILDING: use (cutoff + skin)²
const float neighbor_radius2 = (cutoff + skin) * (cutoff + skin);  // 14.5² = 210.25

float3 dr = apply_pbc(make_float3(
    pos_j.x - pos_i.x,
    pos_j.y - pos_i.y,
    pos_j.z - pos_i.z
));
float r2 = dot3(dr, dr);
if (r2 < neighbor_radius2) { /* add to neighbor list */ }
```

**Force kernel (separate):** Must still apply `r2 < cutoff²` (12² = 144) even when iterating neighbor list:
```cuda
// FORCE KERNEL: use cutoff² (neighbor list may contain pairs up to cutoff+skin)
if (r2 < cutoff2) {
    // compute LJ + Coulomb
}
```

**Verification:** Already uses `apply_pbc()` in force calculation, need to verify in neighbor building.

**Contract:** Neighbor list built to `cutoff + skin` guarantees no missing pairs when atoms move up to `skin/2` between rebuilds. Force kernel filters to actual cutoff.

### Brute-Force Comparison Harness

```rust
#[cfg(test)]
fn verify_neighbor_list_correctness() {
    // For small system (< 500 atoms):
    // 1. Build neighbor list with cell algorithm
    // 2. Build neighbor list with O(N²) brute force
    // 3. Compare: all pairs within cutoff must appear in both
}
```

**Definition of Correctness:**
- Let `NL_cell` = set of pairs from cell-list algorithm
- Let `NL_brute` = set of pairs from O(N²) with PBC minimum image
- For all `(i,j)` where `distance(i,j) < cutoff + skin`: pair MUST appear in `NL_cell`
- False positives OK (pairs slightly beyond cutoff), false negatives NOT OK

### NEIGHBOR_LIST_SIZE Overflow Handling

**Problem:** Fixed `NEIGHBOR_LIST_SIZE` can overflow in dense water (~14.5 Å radius can yield 100+ neighbors).

**Required Implementation:**
```cuda
// In build_neighbor_list kernel:
if (count >= NEIGHBOR_LIST_SIZE) {
    atomicAdd(&d_neighbor_overflow_count, 1);
    d_neighbor_overflow_atoms[atomicAdd(&d_overflow_idx, 1)] = atom_i;
    // Truncate but flag the error
}
```

**Safe Default Sizing:**
- For water at 1.0 g/cm³: ~33 molecules in (14.5 Å)³ sphere
- Each water = 3 atoms → ~100 atom neighbors per atom
- `NEIGHBOR_LIST_SIZE` should be ≥ 128 (conservative) or ≥ 200 (safe)
- Verify at runtime; fail tests if overflow detected

### Files to Modify

| File | Changes | Est. LOC |
|------|---------|----------|
| `amber_mega_fused.rs` | Add skin distance tracking | +40 |
| `amber_mega_fused.rs` | Add displacement-based rebuild | +60 |
| `amber_mega_fused.cu` | Verify PBC in `build_neighbor_list` | +20 |
| `tests/neighbor_list.rs` | Brute-force comparison test | +100 |

### Acceptance Criteria

- [ ] Forces match brute-force O(N²) within 1e-4 kcal/(mol·Å)
- [ ] Displacement-based rebuild triggers correctly
- [ ] Performance scales ~O(N) with water count (not O(N²))
- [ ] All periodic images correctly considered
- [ ] **Overflow detection:** No neighbor list overflow in 216-water box test
- [ ] **Overflow handling:** Tests fail if any overflow detected (not silently truncated)

### Performance Sanity Test

```rust
#[test]
fn test_neighbor_list_performance_baseline() {
    // 216-water box (648 atoms) at 310 K
    // Measure: neighbor list build time (should be < 1 ms on modern GPU)
    // Measure: force computation time per step
    // Record baseline for regression detection
}
```

### Commit

```
feat(gpu): Add displacement-based neighbor list rebuild with PBC verification
```

---

## SECTION 6: PHASE 3 — TIP3P WATER MODEL + CONSTRAINTS

### TIP3P Parameters (AMBER ff14SB Compatible)

| Parameter | Value | Units |
|-----------|-------|-------|
| O–H bond | 0.9572 | Å |
| H–O–H angle | 104.52 | degrees |
| H–H distance | 1.5136 | Å (derived) |
| q(O) | −0.834 | e |
| q(H) | +0.417 | e |
| σ(O) | 3.15061 | Å |
| ε(O) | 0.1521 | kcal/mol |
| Mass(O) | 15.9994 | amu |
| Mass(H) | 1.008 | amu |

### SETTLE Status

**Already implemented:**
- `settle.rs`: Rust wrapper (318 lines)
- `settle.cu`: CUDA kernel (321 lines)
- Batch processing: 1 thread per water
- TIP3P geometry: OH=0.9572Å, HH=1.5136Å

**Verification needed:**
```rust
pub fn check_constraints(&mut self) -> Result<(f32, f32)> {
    // Returns (max_oh_violation, max_hh_violation)
    // Should be < 1e-4 Å after SETTLE
}
```

### Avoid Double-Constraining

H atoms in water should NOT be in H-constraint clusters:
```rust
// In topology preparation:
for cluster in h_clusters {
    if is_water_hydrogen(cluster.hydrogen_atoms) {
        skip; // SETTLE handles these
    }
}
```

### External Topology Support

**Recommended workflow:**
1. Use `tleap` (AmberTools) or `packmol` to solvate protein
2. Export topology JSON with water indices
3. Load in PRISM-4D

**Internal solvation:** Deferred to Phase 3B (optional).

### Files to Modify

| File | Changes | Est. LOC |
|------|---------|----------|
| `settle.rs` | Add constraint violation logging | +20 |
| `generate_ensemble.rs` | Warn if waters in H-clusters | +15 |
| `water_model.rs` (NEW) | TIP3P parameter constants | +50 |

### Acceptance Criteria

- [ ] SETTLE constraint violations < 1e-4 Å
- [ ] No double-constraining of water hydrogens
- [ ] TIP3P parameters match AMBER ff14SB

### Commit

```
feat(physics): Add TIP3P water model parameters and SETTLE verification
```

---

## SECTION 7: PHASE 3B — INTERNAL SOLVATION (REQUIRED FOR SOVEREIGNTY)

**Status:** REQUIRED (removes external tleap/packmol dependency)

### Purpose

Provide a fully self-contained solvation capability so PRISM-4D can prepare explicit solvent systems without external tools. This maintains system sovereignty and simplifies the user workflow.

### Files to Create

| File | Purpose | Est. LOC |
|------|---------|----------|
| `crates/prism-physics/src/solvation.rs` | Core solvation logic | ~350 |
| `crates/prism-physics/src/ions.rs` | Ion parameters + placement | ~100 |
| `tests/solvation_tests.rs` | Validation suite | ~150 |

### Core Data Structures

```rust
// solvation.rs

/// A single TIP3P water molecule with full geometry
pub struct WaterMolecule {
    pub oxygen: [f32; 3],
    pub hydrogen1: [f32; 3],
    pub hydrogen2: [f32; 3],
}

impl WaterMolecule {
    /// Create TIP3P water at specified oxygen position with random orientation
    pub fn tip3p_at(oxygen_pos: [f32; 3], rng: &mut impl Rng) -> Self;

    /// Check if any atom clashes with a point (within min_distance)
    pub fn clashes_with(&self, point: [f32; 3], min_distance: f32) -> bool;
}

/// Ion types supported for neutralization
#[derive(Clone, Copy)]
pub enum IonType {
    Sodium,   // Na+, charge +1
    Chloride, // Cl-, charge -1
}

/// A monatomic ion
pub struct Ion {
    pub ion_type: IonType,
    pub position: [f32; 3],
}

/// Complete solvation box ready for simulation
pub struct SolvationBox {
    /// Original protein atoms (unchanged)
    pub protein_atoms: Vec<Atom>,

    /// Added water molecules
    pub waters: Vec<WaterMolecule>,

    /// Counterions for neutralization
    pub ions: Vec<Ion>,

    /// Orthorhombic box dimensions [Lx, Ly, Lz] in Å
    pub box_dims: [f32; 3],

    /// Indices of water oxygen atoms (for SETTLE)
    pub water_oxygen_indices: Vec<usize>,

    /// Total system charge (should be ~0 after neutralization)
    pub net_charge: f32,
}
```

### Solvation Algorithm (Production-Grade)

```rust
impl SolvationBox {
    /// Create a solvated system from a protein structure
    ///
    /// # Arguments
    /// * `protein` - Protein atoms with positions and charges
    /// * `padding` - Minimum distance from protein to box edge (typically 10-12 Å)
    /// * `min_distance` - Minimum distance from water O to any protein atom (typically 2.5-2.8 Å)
    /// * `rng` - Random number generator for reproducibility
    ///
    /// # Returns
    /// Complete solvation box ready for topology generation
    pub fn from_protein(
        protein: &[Atom],
        padding: f32,
        min_distance: f32,
        rng: &mut impl Rng,
    ) -> Result<Self> {
        // ═══════════════════════════════════════════════════════════════
        // STEP 1: Compute box dimensions from protein bounding box
        // ═══════════════════════════════════════════════════════════════
        let (prot_min, prot_max) = compute_bounding_box(protein);

        // Add padding on all sides
        let box_dims = [
            (prot_max[0] - prot_min[0]) + 2.0 * padding,
            (prot_max[1] - prot_min[1]) + 2.0 * padding,
            (prot_max[2] - prot_min[2]) + 2.0 * padding,
        ];

        // Center protein in box
        let protein_centered = center_in_box(protein, box_dims);

        info!("Box dimensions: {:.1} x {:.1} x {:.1} Å",
              box_dims[0], box_dims[1], box_dims[2]);

        // ═══════════════════════════════════════════════════════════════
        // STEP 2: Build spatial grid for fast clash detection
        // ═══════════════════════════════════════════════════════════════
        // Use cell size slightly larger than min_distance for O(1) lookup
        let cell_size = min_distance + 0.5;
        let protein_grid = SpatialGrid::from_atoms(&protein_centered, cell_size);

        // ═══════════════════════════════════════════════════════════════
        // STEP 3: Place waters on grid with clash rejection
        // ═══════════════════════════════════════════════════════════════
        // Spacing ~3.1 Å gives bulk water density (~0.997 g/cm³)
        let spacing = 3.1_f32;
        let mut waters = Vec::new();

        // Grid placement
        let nx = (box_dims[0] / spacing).ceil() as usize;
        let ny = (box_dims[1] / spacing).ceil() as usize;
        let nz = (box_dims[2] / spacing).ceil() as usize;

        for ix in 0..nx {
            for iy in 0..ny {
                for iz in 0..nz {
                    // Base grid position (centered in cell)
                    let base_pos = [
                        (ix as f32 + 0.5) * spacing,
                        (iy as f32 + 0.5) * spacing,
                        (iz as f32 + 0.5) * spacing,
                    ];

                    // Add random perturbation to avoid crystalline artifacts
                    // ±0.3 Å keeps waters from overlapping while breaking symmetry
                    let perturb = 0.3_f32;
                    let pos = [
                        base_pos[0] + rng.gen_range(-perturb..perturb),
                        base_pos[1] + rng.gen_range(-perturb..perturb),
                        base_pos[2] + rng.gen_range(-perturb..perturb),
                    ];

                    // Skip if outside box (can happen with perturbation at edges)
                    if pos[0] < 0.0 || pos[0] >= box_dims[0] ||
                       pos[1] < 0.0 || pos[1] >= box_dims[1] ||
                       pos[2] < 0.0 || pos[2] >= box_dims[2] {
                        continue;
                    }

                    // CRITICAL: Check distance to protein atoms using spatial grid
                    let min_protein_dist = protein_grid.min_distance_to(pos);
                    if min_protein_dist < min_distance {
                        continue; // Too close to protein
                    }

                    // Create water with random orientation
                    let water = WaterMolecule::tip3p_at(pos, rng);

                    // Verify hydrogen atoms also don't clash
                    let h1_dist = protein_grid.min_distance_to(water.hydrogen1);
                    let h2_dist = protein_grid.min_distance_to(water.hydrogen2);
                    if h1_dist < min_distance * 0.8 || h2_dist < min_distance * 0.8 {
                        continue; // Hydrogen too close
                    }

                    waters.push(water);
                }
            }
        }

        info!("Placed {} water molecules", waters.len());

        // ═══════════════════════════════════════════════════════════════
        // STEP 4: Validate water density
        // ═══════════════════════════════════════════════════════════════
        let volume_nm3 = (box_dims[0] * box_dims[1] * box_dims[2]) / 1000.0;
        let water_mass_g = waters.len() as f32 * 18.015 / 6.022e23;
        let volume_cm3 = volume_nm3 * 1e-21;
        let density = water_mass_g / volume_cm3;

        // Account for protein volume displacement
        let protein_volume_approx = protein.len() as f32 * 20.0; // ~20 Å³ per atom
        let water_volume = (box_dims[0] * box_dims[1] * box_dims[2]) - protein_volume_approx;
        let effective_density = (waters.len() as f32 * 30.0) / water_volume; // 30 Å³ per water

        if effective_density < 0.028 || effective_density > 0.038 {
            // Expected: ~0.033 waters/Å³ (bulk water)
            warn!("Water density {:.4} waters/Å³ outside normal range [0.028, 0.038]",
                  effective_density);
        }

        // ═══════════════════════════════════════════════════════════════
        // STEP 5: Add counterions for charge neutralization
        // ═══════════════════════════════════════════════════════════════
        let protein_charge: f32 = protein.iter().map(|a| a.charge).sum();
        let ions = place_counterions(&mut waters, protein_charge, &protein_grid, rng)?;

        let ion_charge: f32 = ions.iter().map(|i| i.charge()).sum();
        let net_charge = protein_charge + ion_charge;

        info!("Added {} ions (protein charge: {:.1}, net charge: {:.2})",
              ions.len(), protein_charge, net_charge);

        if net_charge.abs() > 0.1 {
            warn!("System not fully neutralized: net charge = {:.2}", net_charge);
        }

        // ═══════════════════════════════════════════════════════════════
        // STEP 6: Compute water oxygen indices for SETTLE
        // ═══════════════════════════════════════════════════════════════
        // Atom ordering: [protein atoms...][water O, H, H][water O, H, H]...[ions...]
        let protein_atom_count = protein.len();
        let water_oxygen_indices: Vec<usize> = (0..waters.len())
            .map(|i| protein_atom_count + i * 3) // O is first atom of each water
            .collect();

        Ok(Self {
            protein_atoms: protein_centered,
            waters,
            ions,
            box_dims,
            water_oxygen_indices,
            net_charge,
        })
    }

    /// Convert to topology for simulation
    pub fn to_topology(&self) -> AmberTopology {
        let mut atoms = Vec::new();
        let mut bonds = Vec::new();
        let mut angles = Vec::new();

        // Add protein atoms (unchanged)
        atoms.extend(self.protein_atoms.iter().cloned());

        // Add water atoms with TIP3P parameters
        for water in &self.waters {
            let o_idx = atoms.len();

            // Oxygen
            atoms.push(Atom {
                name: "OW".to_string(),
                position: water.oxygen,
                mass: 15.9994,
                charge: -0.834,
                atom_type: "OW".to_string(),
                sigma: 3.15061,
                epsilon: 0.1521,
            });

            // Hydrogen 1
            atoms.push(Atom {
                name: "HW1".to_string(),
                position: water.hydrogen1,
                mass: 1.008,
                charge: 0.417,
                atom_type: "HW".to_string(),
                sigma: 0.0,
                epsilon: 0.0,
            });

            // Hydrogen 2
            atoms.push(Atom {
                name: "HW2".to_string(),
                position: water.hydrogen2,
                mass: 1.008,
                charge: 0.417,
                atom_type: "HW".to_string(),
                sigma: 0.0,
                epsilon: 0.0,
            });

            // O-H bonds (for bookkeeping; SETTLE handles constraints)
            bonds.push(Bond { i: o_idx, j: o_idx + 1, r0: 0.9572, k: 553.0 });
            bonds.push(Bond { i: o_idx, j: o_idx + 2, r0: 0.9572, k: 553.0 });

            // H-O-H angle
            angles.push(Angle {
                i: o_idx + 1,
                j: o_idx,
                k: o_idx + 2,
                theta0: 104.52_f32.to_radians(),
                k: 100.0,
            });
        }

        // Add ions
        for ion in &self.ions {
            let (name, mass, charge, sigma, epsilon) = match ion.ion_type {
                IonType::Sodium => ("Na+", 22.99, 1.0, 2.43, 0.0874),
                IonType::Chloride => ("Cl-", 35.45, -1.0, 4.40, 0.1),
            };
            atoms.push(Atom {
                name: name.to_string(),
                position: ion.position,
                mass,
                charge,
                atom_type: name.to_string(),
                sigma,
                epsilon,
            });
        }

        AmberTopology {
            atoms,
            bonds,
            angles,
            dihedrals: vec![], // Waters have no dihedrals
            box_vectors: Some(self.box_dims),
            water_oxygens: self.water_oxygen_indices.clone(),
            // ... other fields
        }
    }
}
```

### Water Molecule Geometry

```rust
impl WaterMolecule {
    /// TIP3P geometry constants
    const OH_BOND: f32 = 0.9572;      // Å
    const HOH_ANGLE: f32 = 104.52;    // degrees
    const HH_DISTANCE: f32 = 1.5136;  // Å (derived)

    /// Create TIP3P water at oxygen position with random orientation
    pub fn tip3p_at(oxygen: [f32; 3], rng: &mut impl Rng) -> Self {
        // Generate random rotation (uniform on SO(3))
        let rotation = random_rotation_matrix(rng);

        // TIP3P geometry in local frame (O at origin)
        // H atoms at ±52.26° from y-axis in xy-plane
        let half_angle = (Self::HOH_ANGLE / 2.0).to_radians();
        let h1_local = [
            Self::OH_BOND * half_angle.sin(),
            Self::OH_BOND * half_angle.cos(),
            0.0,
        ];
        let h2_local = [
            -Self::OH_BOND * half_angle.sin(),
            Self::OH_BOND * half_angle.cos(),
            0.0,
        ];

        // Rotate and translate to final position
        let h1_rotated = rotate_vector(h1_local, &rotation);
        let h2_rotated = rotate_vector(h2_local, &rotation);

        Self {
            oxygen,
            hydrogen1: [
                oxygen[0] + h1_rotated[0],
                oxygen[1] + h1_rotated[1],
                oxygen[2] + h1_rotated[2],
            ],
            hydrogen2: [
                oxygen[0] + h2_rotated[0],
                oxygen[1] + h2_rotated[1],
                oxygen[2] + h2_rotated[2],
            ],
        }
    }

    /// Check geometry validity
    pub fn validate(&self) -> Result<()> {
        let oh1 = distance(self.oxygen, self.hydrogen1);
        let oh2 = distance(self.oxygen, self.hydrogen2);
        let hh = distance(self.hydrogen1, self.hydrogen2);

        let oh_tol = 1e-4;
        let hh_tol = 1e-4;

        if (oh1 - Self::OH_BOND).abs() > oh_tol {
            bail!("O-H1 distance {:.4} != {:.4}", oh1, Self::OH_BOND);
        }
        if (oh2 - Self::OH_BOND).abs() > oh_tol {
            bail!("O-H2 distance {:.4} != {:.4}", oh2, Self::OH_BOND);
        }
        if (hh - Self::HH_DISTANCE).abs() > hh_tol {
            bail!("H-H distance {:.4} != {:.4}", hh, Self::HH_DISTANCE);
        }
        Ok(())
    }
}

/// Generate uniform random rotation matrix (Arvo's method)
fn random_rotation_matrix(rng: &mut impl Rng) -> [[f32; 3]; 3] {
    let u1: f32 = rng.gen();
    let u2: f32 = rng.gen();
    let u3: f32 = rng.gen();

    let theta = 2.0 * std::f32::consts::PI * u1;
    let phi = 2.0 * std::f32::consts::PI * u2;
    let z = u3;

    let r = z.sqrt();
    let v = [r * phi.cos(), r * phi.sin(), (1.0 - z).sqrt()];
    let s = theta.sin();
    let c = theta.cos();

    // Householder matrix
    [
        [2.0 * v[0] * v[0] - 1.0, 2.0 * v[0] * v[1] - 2.0 * v[2] * s, 2.0 * v[0] * v[2] + 2.0 * v[1] * s],
        [2.0 * v[0] * v[1] + 2.0 * v[2] * s, 2.0 * v[1] * v[1] - 1.0, 2.0 * v[1] * v[2] - 2.0 * v[0] * s],
        [2.0 * v[0] * v[2] - 2.0 * v[1] * s, 2.0 * v[1] * v[2] + 2.0 * v[0] * s, 2.0 * v[2] * v[2] - 1.0],
    ]
}
```

### Counterion Placement

```rust
// ions.rs

impl Ion {
    pub fn charge(&self) -> f32 {
        match self.ion_type {
            IonType::Sodium => 1.0,
            IonType::Chloride => -1.0,
        }
    }
}

/// Place counterions to neutralize system charge
///
/// Strategy: Replace waters that are far from the protein with ions.
/// This mimics how tleap/GROMACS place ions.
fn place_counterions(
    waters: &mut Vec<WaterMolecule>,
    protein_charge: f32,
    protein_grid: &SpatialGrid,
    rng: &mut impl Rng,
) -> Result<Vec<Ion>> {
    let n_ions = protein_charge.abs().round() as usize;

    if n_ions == 0 {
        info!("Protein is neutral, no counterions needed");
        return Ok(vec![]);
    }

    let ion_type = if protein_charge < 0.0 {
        IonType::Sodium  // Add positive ions to neutralize negative protein
    } else {
        IonType::Chloride // Add negative ions to neutralize positive protein
    };

    // Score waters by distance from protein (prefer replacing distant waters)
    let mut water_scores: Vec<(usize, f32)> = waters.iter()
        .enumerate()
        .map(|(i, w)| {
            let dist = protein_grid.min_distance_to(w.oxygen);
            // Add small random component to break ties
            let score = dist + rng.gen_range(0.0..1.0);
            (i, score)
        })
        .collect();

    // Sort by distance (descending) - furthest waters first
    water_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Also ensure ions aren't too close to each other (min 5 Å apart)
    let mut ion_positions: Vec<[f32; 3]> = Vec::new();
    let mut ions = Vec::new();
    let mut indices_to_remove = Vec::new();

    for (idx, _score) in water_scores.iter() {
        if ions.len() >= n_ions {
            break;
        }

        let water_pos = waters[*idx].oxygen;

        // Check distance to existing ions
        let too_close = ion_positions.iter()
            .any(|pos| distance(*pos, water_pos) < 5.0);

        if too_close {
            continue;
        }

        ions.push(Ion {
            ion_type,
            position: water_pos,
        });
        ion_positions.push(water_pos);
        indices_to_remove.push(*idx);
    }

    if ions.len() < n_ions {
        warn!("Could only place {} of {} required ions", ions.len(), n_ions);
    }

    // Remove waters that were replaced by ions (in reverse order to preserve indices)
    indices_to_remove.sort_by(|a, b| b.cmp(a));
    for idx in indices_to_remove {
        waters.remove(idx);
    }

    Ok(ions)
}
```

### Spatial Grid for Fast Clash Detection

```rust
/// Spatial grid for O(1) nearest-neighbor queries
struct SpatialGrid {
    cells: HashMap<(i32, i32, i32), Vec<[f32; 3]>>,
    cell_size: f32,
}

impl SpatialGrid {
    fn from_atoms(atoms: &[Atom], cell_size: f32) -> Self {
        let mut cells: HashMap<(i32, i32, i32), Vec<[f32; 3]>> = HashMap::new();

        for atom in atoms {
            let key = Self::cell_key(atom.position, cell_size);
            cells.entry(key).or_default().push(atom.position);
        }

        Self { cells, cell_size }
    }

    fn cell_key(pos: [f32; 3], cell_size: f32) -> (i32, i32, i32) {
        (
            (pos[0] / cell_size).floor() as i32,
            (pos[1] / cell_size).floor() as i32,
            (pos[2] / cell_size).floor() as i32,
        )
    }

    /// Find minimum distance from query point to any atom in grid
    fn min_distance_to(&self, query: [f32; 3]) -> f32 {
        let (cx, cy, cz) = Self::cell_key(query, self.cell_size);

        let mut min_dist = f32::MAX;

        // Check 3x3x3 neighborhood of cells
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let key = (cx + dx, cy + dy, cz + dz);
                    if let Some(atoms) = self.cells.get(&key) {
                        for atom_pos in atoms {
                            let d = distance(query, *atom_pos);
                            if d < min_dist {
                                min_dist = d;
                            }
                        }
                    }
                }
            }
        }

        min_dist
    }
}

fn distance(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}
```

### Acceptance Criteria

| Test | Criterion |
|------|-----------|
| Water count | Within 10% of expected for box volume |
| Water density | 0.028–0.038 waters/Å³ (accounting for protein) |
| Net charge | \|net_charge\| < 0.1 e after neutralization |
| Min water-protein distance | All O atoms > min_distance from protein |
| Min water-water distance | All O-O pairs > 2.4 Å |
| Water geometry | O-H = 0.9572 ± 0.0001 Å, H-H = 1.5136 ± 0.0001 Å |
| Ion spacing | All ion pairs > 5.0 Å apart |
| Reproducibility | Same RNG seed → identical output |

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_water_geometry() {
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..1000 {
            let pos = [rng.gen_range(0.0..10.0), rng.gen_range(0.0..10.0), rng.gen_range(0.0..10.0)];
            let water = WaterMolecule::tip3p_at(pos, &mut rng);
            water.validate().unwrap();
        }
    }

    #[test]
    fn test_solvation_density() {
        // Small test protein (alanine dipeptide, 22 atoms)
        let protein = load_test_protein("ala_dipeptide.pdb");
        let mut rng = StdRng::seed_from_u64(12345);

        let solvbox = SolvationBox::from_protein(&protein, 10.0, 2.5, &mut rng).unwrap();

        // Check density
        let vol = solvbox.box_dims[0] * solvbox.box_dims[1] * solvbox.box_dims[2];
        let waters_per_A3 = solvbox.waters.len() as f32 / vol;
        assert!(waters_per_A3 > 0.025 && waters_per_A3 < 0.040,
                "Water density {} outside expected range", waters_per_A3);
    }

    #[test]
    fn test_no_clashes() {
        let protein = load_test_protein("1ubq.pdb");
        let mut rng = StdRng::seed_from_u64(42);
        let min_distance = 2.5;

        let solvbox = SolvationBox::from_protein(&protein, 10.0, min_distance, &mut rng).unwrap();

        // Verify no water oxygen is within min_distance of any protein atom
        for water in &solvbox.waters {
            for atom in &solvbox.protein_atoms {
                let d = distance(water.oxygen, atom.position);
                assert!(d >= min_distance - 0.01,
                        "Water at {:?} too close to protein atom at {:?}: {:.3} Å",
                        water.oxygen, atom.position, d);
            }
        }
    }

    #[test]
    fn test_neutralization() {
        // Protein with +5 charge
        let mut protein = load_test_protein("charged_protein.pdb");
        // Verify charge
        let charge: f32 = protein.iter().map(|a| a.charge).sum();

        let mut rng = StdRng::seed_from_u64(42);
        let solvbox = SolvationBox::from_protein(&protein, 10.0, 2.5, &mut rng).unwrap();

        assert!(solvbox.net_charge.abs() < 0.1,
                "System not neutralized: net charge = {}", solvbox.net_charge);
    }

    #[test]
    fn test_reproducibility() {
        let protein = load_test_protein("1ubq.pdb");

        // Same seed should give identical results
        let mut rng1 = StdRng::seed_from_u64(42);
        let mut rng2 = StdRng::seed_from_u64(42);

        let box1 = SolvationBox::from_protein(&protein, 10.0, 2.5, &mut rng1).unwrap();
        let box2 = SolvationBox::from_protein(&protein, 10.0, 2.5, &mut rng2).unwrap();

        assert_eq!(box1.waters.len(), box2.waters.len());
        assert_eq!(box1.ions.len(), box2.ions.len());

        for (w1, w2) in box1.waters.iter().zip(box2.waters.iter()) {
            assert_eq!(w1.oxygen, w2.oxygen);
        }
    }
}
```

### CLI Integration

```rust
// In generate_ensemble.rs

#[derive(Parser)]
struct Args {
    /// Input PDB file (protein only)
    #[arg(long)]
    pdb: PathBuf,

    /// Solvent model
    #[arg(long, default_value = "implicit")]
    solvent: SolventModel,

    /// Box padding for explicit solvent (Å)
    #[arg(long, default_value = "12.0")]
    box_padding: f32,

    /// Minimum water-protein distance (Å)
    #[arg(long, default_value = "2.5")]
    water_min_distance: f32,

    /// Random seed for reproducible solvation
    #[arg(long, default_value = "42")]
    solvation_seed: u64,
}

// In main():
if args.solvent == SolventModel::Explicit {
    info!("Solvating protein with {} Å padding...", args.box_padding);

    let mut rng = StdRng::seed_from_u64(args.solvation_seed);
    let solvbox = SolvationBox::from_protein(
        &protein_atoms,
        args.box_padding,
        args.water_min_distance,
        &mut rng,
    )?;

    info!("Created solvation box: {} waters, {} ions, box = {:?}",
          solvbox.waters.len(), solvbox.ions.len(), solvbox.box_dims);

    let topology = solvbox.to_topology();
    // ... proceed with simulation
}
```

### Commit

```
feat(physics): Add internal solvation with TIP3P water placement and ion neutralization
```

---

## SECTION 8: PHASE 4 — PME ELECTROSTATICS (WITH HARNESS)

### PME Validation Harness (REQUIRED)

**Before integrating PME into protein simulations:**

```rust
#[test]
fn test_pme_vs_direct_sum() {
    // Small system: 2 charges in periodic box
    // Compare PME energy to Ewald direct sum
    // Tolerance: < 0.1 kcal/mol
}

#[test]
fn test_pme_force_accuracy() {
    // Numerical gradient vs analytical PME force
    // Tolerance: < 1e-3 kcal/(mol·Å)
}

#[test]
fn test_pme_water_box() {
    // 216 TIP3P waters in 18.6 Å box
    // Run 100 ps, verify temperature stable
}
```

### PME Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Grid spacing | 1.0 Å | `PME_GRID_SPACING` |
| B-spline order | 4 | Standard for MD |
| Ewald tolerance | 1e-5 | Relative error tolerance |
| Ewald β | **COMPUTED** | `β = sqrt(-ln(tol)) / cutoff` ≈ 0.283 Å⁻¹ |
| Real-space cutoff | 12 Å | Same as NB cutoff |

**IMPORTANT:** β must be computed at runtime from `(tolerance, cutoff)`, not hardcoded.
```rust
let beta = compute_ewald_beta(self.cutoff, self.pme_tolerance);
```

### CUDA Stages

1. `pme_zero_grid` — Clear charge grid
2. `pme_spread_charges` — B-spline interpolation
3. `cufftExecR2C` — Forward FFT
4. `pme_reciprocal_convolution` — Green's function
5. `cufftExecC2R` — Inverse FFT
6. `pme_interpolate_forces` — Back-interpolate
7. `pme_self_energy` — Self-energy correction

### Files to Modify

| File | Changes | Est. LOC |
|------|---------|----------|
| `tests/pme_validation.rs` (NEW) | PME harness tests | +200 |
| `pme.rs` | Add debug logging, energy return | +30 |

### Acceptance Criteria

- [ ] PME matches direct Ewald sum within 0.1 kcal/mol
- [ ] Forces match numerical gradient within 1e-3
- [ ] 216-water box stable for 100 ps
- [ ] Temperature remains 310 ± 20 K

### Abort Criteria (Before Protein Integration)

If any of these fail, do not proceed to protein runs:
1. Energy error > 1 kcal/mol vs reference
2. Water box temperature drifts > 50 K
3. NaN or Inf in forces

### Commit

```
test(gpu): Add PME validation harness with direct sum comparison
```

---

## SECTION 8B: MODE OWNERSHIP TABLE

This table prevents "mixed-mode defaults" bugs by explicitly showing which features are owned by each mode:

| Toggle | Implicit | Explicit |
|--------|----------|----------|
| PBC | OFF | ON |
| PME | OFF | ON |
| SETTLE | OFF | ON (if waters present) |
| H-constraints | Optional | ON (protein only) |
| COM removal | OFF | ON |
| Restraints default | ON | OFF |
| Dielectric | ε = 4r (screened) | ε = 1 (explicit water) |

**Usage:** When switching `--solvent`, all toggles in this table flip to their column's value. No manual override should be needed for typical workflows.

---

## SECTION 9: PHASE 5 — CLI INTEGRATION

### New CLI Arguments

**Primary Input Contract:**
- `--topology <PATH>` is the PRIMARY input (prepared JSON from `prepare_protein.py`)
- `--pdb <PATH>` is OPTIONAL (used only for atom naming in output)
- For explicit solvent, the topology JSON must include `box_vectors` and `water_oxygens`

```rust
#[derive(Clone, ValueEnum)]
pub enum SolventModel {
    Implicit,
    Explicit,
}

#[derive(Parser)]
struct Args {
    /// Primary input: prepared topology JSON (REQUIRED)
    #[arg(long)]
    topology: PathBuf,

    /// Optional: original PDB for atom names in output
    #[arg(long)]
    pdb: Option<PathBuf>,

    #[arg(long, default_value = "implicit")]
    solvent: SolventModel,

    #[arg(long, default_value = "12.0")]
    box_padding: f32,  // Å, for solvation

    #[arg(long, default_value = "2.5")]
    neighbor_skin: f32,  // Å, for Verlet lists

    #[arg(long, default_value = "1.0")]
    pme_grid_spacing: f32,  // Å

    #[arg(long, default_value = "0.0")]
    restraints_k: f32,  // kcal/(mol·Å²), 0 = disabled

    #[arg(long)]
    energy_log: Option<PathBuf>,

    #[arg(long)]
    temperature_log: Option<PathBuf>,
}
```

**Validation for Explicit Solvent:**
```rust
if args.solvent == SolventModel::Explicit {
    if topology.box_vectors.is_none() {
        bail!("Explicit solvent requires box_vectors in topology JSON");
    }
    if topology.water_oxygens.is_empty() {
        warn!("No water molecules found; running explicit solvent without water");
    }
}
```

### Explicit Solvent Defaults

```rust
if args.solvent == SolventModel::Explicit {
    // PBC: ON (required)
    hmc.set_pbc_box(box_dims)?;

    // PME: ON
    hmc.enable_explicit_solvent(box_dims)?;

    // SETTLE: ON (if waters present)
    hmc.set_water_molecules(&water_oxygens)?;

    // Restraints: OFF by default, but allowed for early equilibration
    if args.restraints_k > 0.0 {
        info!("Explicit solvent with restraints k={} (OK for equilibration)", args.restraints_k);
        hmc.set_position_restraints(&heavy_atoms, args.restraints_k)?;
    }

    // COM removal: ON
    hmc.enable_com_removal(true);
}
```

### Files to Modify

| File | Changes | Est. LOC |
|------|---------|----------|
| `generate_ensemble.rs` | Add CLI args, mode switching | +150 |
| `amber_mega_fused.rs` | Add `enable_com_removal()` | +20 |

### Acceptance Criteria

- [ ] `--solvent implicit` produces identical results to current
- [ ] `--solvent explicit` enables PBC, PME, SETTLE
- [ ] Restraints default OFF for explicit, ON for implicit
- [ ] Energy logs written when requested

### Commit

```
feat(validation): Add --solvent explicit mode with auto-defaults
```

---

## SECTION 10: PHASE 6 — TIERED VERIFICATION

### Tier 1: Unit Tests (Automated)

| Test | Purpose | Pass Criteria |
|------|---------|---------------|
| `test_minimum_image` | PBC wrapping | Distances < L/2 |
| `test_neighbor_list_pbc` | Neighbors across boundary | Matches brute-force |
| `test_settle_tolerance` | Constraint satisfaction | < 1e-4 Å violation |
| `test_pme_vs_ewald` | PME accuracy | < 0.1 kcal/mol error |

### Tier 2: Short Simulations (Semi-Automated)

| Test | System | Duration | Pass Criteria |
|------|--------|----------|---------------|
| Pure water | 216 TIP3P | 100–500 ps | T = 310±15 K, stable |
| Small protein | 1UBQ + 1000 waters | 1 ns | RMSD < 3 Å |

### Tier 3: Production Validation (Manual)

| Test | System | Duration | Pass Criteria |
|------|--------|----------|---------------|
| RBD solvated | 6M0J + waters | 5–10 ns | Fold maintained |
| Long stability | Any | 10+ ns | No drift, T stable |

### Verification Commands

```bash
# Tier 1
cargo test -p prism-gpu pbc
cargo test -p prism-gpu neighbor
cargo test -p prism-gpu settle
cargo test -p prism-gpu pme

# Tier 2
cargo run --release -p prism-validation --bin generate-ensemble -- \
    --topology water_box.json \
    --solvent explicit \
    --steps 250000 \
    --temperature 310 \
    --energy-log tier2_water.csv

# Tier 3
cargo run --release -p prism-validation --bin generate-ensemble -- \
    --topology 6m0j_solvated.json \
    --solvent explicit \
    --steps 5000000 \
    --save-interval 1000 \
    --energy-log tier3_rbd.csv
```

---

## SECTION 11: COMPARISON TABLE (IMPLICIT vs EXPLICIT)

| Metric | Implicit (Baseline) | Explicit (Target) |
|--------|---------------------|-------------------|
| RMSD from native | ~1.0 Å | < 3.0 Å |
| RMSF mean | ~0.4 Å | 0.5–1.5 Å |
| Temperature | 310 ± 15 K | 310 ± 10 K |
| Energy | N/A | Bounded (no explosion) |
| Stability | Yes | Yes |
| Fold maintained | Yes (restrained) | Yes (realistic environment) |

**Note:** Explicit solvent provides a realistic aqueous environment. Larger RMSD/RMSF fluctuations are expected without restraints. Gentle equilibration restraints are acceptable if needed.

---

## SECTION 12: FILE SUMMARY TABLE

| File | Action | Est. LOC | Phase |
|------|--------|----------|-------|
| `amber_mega_fused.rs` | Modify | +200 | 0, 1, 2, 5 |
| `amber_mega_fused.cu` | Modify | +100 | 1, 2 |
| `generate_ensemble.rs` | Modify | +250 | 0, 5 |
| `water_model.rs` | Create | 50 | 3 |
| `solvation.rs` | Create | 350 | 3B |
| `ions.rs` | Create | 100 | 3B |
| `tests/solvation_tests.rs` | Create | 150 | 3B |
| `tests/pme_validation.rs` | Create | 200 | 4 |
| `tests/neighbor_list.rs` | Create | 100 | 2 |
| `settle.rs` | Modify | +20 | 3 |
| `pme.rs` | Modify | +30 | 4 |
| **TOTAL** | | ~1550 | |

---

## SECTION 13: RISK ANALYSIS (MANDATORY)

### Top 5 Technical Risks

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|------------|--------|------------|
| 1 | PME force errors cause instability | Medium | High | Validation harness before protein runs |
| 2 | Neighbor list misses pairs at boundaries | Medium | High | Brute-force comparison test |
| 3 | SETTLE + H-constraints conflict | Low | Medium | Exclude water H from H-clusters |
| 4 | Temperature drift with constraints | Medium | Medium | Proper DOF accounting |
| 5 | Performance regression | Low | Low | Benchmark before/after |

### Risk Details

**Risk 1: PME Force Errors**
- *Symptom:* Energy explosion, NaN forces, protein unfolds
- *Mitigation:* Tier 1 PME harness must pass before any protein integration
- *Abort criteria:* PME error > 1 kcal/mol on 2-charge test

**Risk 2: Neighbor List Boundary Errors**
- *Symptom:* Missing forces, asymmetric behavior at box edges
- *Mitigation:* Brute-force comparison on small system
- *Abort criteria:* Any pair within cutoff missing from neighbor list

**Risk 3: Double-Constraining**
- *Symptom:* Water geometry distortion, energy artifacts
- *Mitigation:* Topology validation warns if water H in H-clusters
- *Detection:* SETTLE violation check

**Risk 4: Temperature Drift**
- *Symptom:* Temperature slowly rises/falls from 310 K
- *Mitigation:* Proper DOF = 3N - 3 - 3*waters - h_constraints
- *Detection:* Temperature log shows trend

**Risk 5: Performance Regression**
- *Symptom:* Slower than current implicit solvent
- *Mitigation:* Benchmark ns/day before and after
- *Acceptable:* 2x slower is OK (explicit has more atoms)

### Abort Criteria Before PME Integration

Do NOT proceed to Phase 4 (PME) if:
1. ❌ Neighbor list fails brute-force comparison
2. ❌ Position wrapping fails single-particle test
3. ❌ SETTLE violations > 1e-3 Å
4. ❌ Temperature calculation gives wrong result

---

## SECTION 14: IMPLEMENTATION ORDER (LOCKED)

| Order | Phase | Description | Commit Message |
|-------|-------|-------------|----------------|
| 0 | Diagnostics | Energy/temperature logging + DOF | `feat(simulation): Add energy and temperature logging` |
| 0.5 | Regression | Save implicit solvent reference trajectory | (no commit - test artifact) |
| 1 | PBC Core | Position wrapping + COM removal | `feat(gpu): Add PBC position wrapping and COM drift removal` |
| 2 | Neighbor Lists | Displacement-based rebuild + verification | `feat(gpu): Add displacement-based neighbor list rebuild` |
| 3 | TIP3P | Water model + SETTLE verification | `feat(physics): Add TIP3P water model parameters` |
| 3B | Solvation | Internal solvation + ion neutralization | `feat(physics): Add internal solvation with TIP3P water placement` |
| 4 | PME | Validation harness + integration | `test(gpu): Add PME validation harness` |
| 5 | CLI | --solvent flag + auto-defaults | `feat(validation): Add --solvent explicit mode` |
| 6 | Verification | Tiered testing | `test(validation): Add explicit solvent verification suite` |

---

## EXECUTION CHECKLIST

After each phase, verify before proceeding:

- [ ] **Phase 0:** Energy CSV matches expected format, T~310K
- [ ] **Phase 0.5:** Reference implicit trajectory saved, checksums recorded
- [ ] **Phase 1:** Single particle wraps correctly, COM drift = 0; implicit regression passes
- [ ] **Phase 2:** Forces match brute-force, performance OK
- [ ] **Phase 3:** SETTLE violations < 1e-4 Å
- [ ] **Phase 3B:** Solvation produces correct density, neutralized system, no clashes
- [ ] **Phase 4:** PME harness passes all tests
- [ ] **Phase 5:** `--solvent implicit` identical to current
- [ ] **Phase 6:** Tier 2 tests pass (water box stable)

---

## PATH TO 100+ ns STABILITY

**Not a guarantee, but a pathway:**

1. Complete all phases with verification
2. Run Tier 2 tests (1 ns small protein)
3. Run Tier 3 tests (5-10 ns RBD)
4. If stable, extend to 100 ns with monitoring
5. If instabilities:
   - Check energy log for drift
   - Verify SETTLE constraints
   - Check neighbor list freshness
   - Consider smaller timestep (1.5 fs)

**Success criteria for 100 ns:**
- Temperature: 310 ± 10 K throughout
- Energy: Bounded, no monotonic drift
- RMSD: < 5 Å from initial (protein fold maintained)
- No NaN/Inf values
