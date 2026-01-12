# CLAUDE.md - PRISM4D Project Instructions

This file contains instructions for Claude Code when working on this project.

## ACTIVE IMPLEMENTATION: Phase 6 SOTA Cryptic Site Detection

**Mandatory Reference**: Before ANY implementation work, Claude MUST read and follow:
- `results/phase6_sota_plan.md` (v2.0 - the authoritative specification)

## Non-Negotiable Constraints

### 1. Zero Fallback Policy
```
FORBIDDEN:
- Silent CPU fallback when GPU unavailable
- Mock implementations or placeholder returns
- todo!() or unimplemented!() in production code
- Hardcoded return values
- Skipping GPU initialization errors

REQUIRED:
- Explicit error on missing GPU
- Real computed values from actual GPU operations
- All functions must process actual input data
```

### 2. Zero External Dependencies
```
FORBIDDEN:
- PyTorch, TensorFlow, or external ML frameworks
- AlphaFlow, ESM-2, or pre-trained models
- Any dependency not already in Cargo.toml

REQUIRED:
- Native Rust/CUDA implementations only
- PRISM-ZrO (SNN + RLS) for learning
- PRISM-NOVA (HMC + AMBER) for sampling
- Blake3 hashing for integrity/caching
- Betti numbers (beta_0, beta_1, beta_2) for TDA
```

### 2.1 Core PRISM Features (MAINTAINED ALL PHASES)
```
BETTI NUMBERS (TDA Topology):
- beta_0: Connected components
- beta_1: Loops/tunnels (pocket indicators)
- beta_2: Voids/cavities (cryptic site signatures)
- Computed via prism_gpu::tda module
- GPU-accelerated alpha complex filtration

BLAKE3 HASHING (Integrity & Caching):
- Structure fingerprinting (unique PDB identification)
- Conformation cache keys (avoid recomputation)
- Result integrity verification
- Reproducibility checksums for all outputs

These features are MAINTAINED throughout all phases.
Phase 7 EXTENDS Betti to full persistence diagrams.
```

### 3. Hybrid Sampler Architecture (CRITICAL)

PRISM has full-atom AMBER ff14SB with bonds, angles, dihedrals already implemented.
Phase 6 uses a HYBRID approach for conformational sampling:

```
┌─────────────────────────────────────────────────────────────┐
│                  HYBRID SAMPLING ROUTER                      │
├─────────────────────────────────────────────────────────────┤
│  Input: Sanitized Structure                                  │
│              │                                               │
│              ▼                                               │
│     ┌────────────────────┐                                  │
│     │ n_atoms <= 512 ?   │                                  │
│     └────────────────────┘                                  │
│         │           │                                        │
│        YES          NO                                       │
│         │           │                                        │
│         ▼           ▼                                        │
│  ┌────────────┐  ┌─────────────────┐                        │
│  │ PrismNova  │  │ AmberMegaFused  │                        │
│  │            │  │                 │                        │
│  │ + TDA β₂   │  │ Full MD engine  │                        │
│  │ + Active   │  │ (no TDA, but    │                        │
│  │   Inference│  │  proven AMBER)  │                        │
│  └────────────┘  └─────────────────┘                        │
│         │           │                                        │
│         └─────┬─────┘                                        │
│               ▼                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │         Unified Conformation Output                  │    │
│  │         (Same format regardless of backend)          │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘

NOVA_ATOM_LIMIT = 512 (due to shared memory constraints)

Coverage:
- ~40% of CryptoBench: PrismNova (TDA + Active Inference)
- ~60% of CryptoBench: AmberMegaFusedHmc (full MD, no TDA)
- 100% of structures processable

Existing AMBER Components (DO NOT REIMPLEMENT):
- prism_gpu::AmberMegaFusedHmc    - Full HMC with O(N) neighbor lists
- prism_gpu::AmberBondedForces    - GPU bond/angle/dihedral forces
- prism_physics::amber_topology   - Topology generator from PDB
- prism_gpu::amber_bonded.cu      - CUDA kernels for bonded terms
```

### 4. Zero Data Leakage
```
FORBIDDEN:
- Using test set structures during training
- Mixing train/test in any validation
- Accessing ground truth before prediction

REQUIRED:
- Strict 885/222 train/test split
- All metrics computed on test set only
```

### 5. Parallel Implementation Architecture (CRITICAL)

Phase 6 uses a **side-by-side implementation with shadow pipeline** pattern:

```
ARCHITECTURE:
- NOVA Path (Greenfield): TDA + Active Inference, ≤512 atoms
- AMBER Path (Stable): Proven MD, no atom limit
- Shadow Pipeline: Run both, compare outputs, validate before promotion
- Router: Selects path based on structure size and migration stage

ISOLATION RULES:
- nova_path.rs MUST NEVER import from amber_path.rs
- amber_path.rs MUST NEVER import from nova_path.rs
- Both MUST implement SamplingBackend trait exactly
- Changes to contract.rs require updates to BOTH paths

MIGRATION STAGES (Strangler Pattern):
  StableOnly → Shadow → Canary(10%) → Canary(50%) → GreenfieldPrimary → GreenfieldOnly
       ↑___________________________________________↓ (auto-rollback on failures)

THE CONTRACT (sampling/contract.rs):
  trait SamplingBackend {
    fn id(&self) -> BackendId;
    fn capabilities(&self) -> BackendCapabilities;
    fn load_structure(&mut self, structure: &SanitizedStructure) -> Result<()>;
    fn sample(&mut self, config: &SamplingConfig) -> Result<SamplingResult>;
    fn reset(&mut self) -> Result<()>;
    fn estimate_vram_mb(&self, n_atoms: usize) -> f32;
  }
```

Key Files for Parallel Implementation:
```
sampling/contract.rs        - THE LAW (trait definition)
sampling/paths/nova_path.rs - Greenfield implementation
sampling/paths/amber_path.rs - Stable implementation
sampling/shadow/comparator.rs - Output comparison
sampling/migration/feature_flags.rs - Rollout control
sampling/router/mod.rs      - Entry point
```

## Phase 6 Implementation Order

Claude MUST implement in this EXACT order:

### Week 0 (PREREQUISITE - Must complete first)
1. Verify environment (rustc 1.75+, nvcc 12.0+)
2. Run `cargo check -p prism-gpu --features cuda`
3. Download CryptoBench dataset (1107 structures)
4. Download apo-holo pairs (15 pairs)
5. Create baseline documentation

### Weeks 1-2: GPU SNN Scale-Up
Files to create IN ORDER:
1. `crates/prism-validation/src/cryptic_features.rs`
2. `crates/prism-validation/src/gpu_zro_cryptic_scorer.rs`
3. `crates/prism-validation/src/tests/gpu_scorer_tests.rs`

CHECKPOINT: Must pass before proceeding:
```bash
cargo test --release -p prism-validation --features cuda gpu_scorer
CUDA_VISIBLE_DEVICES="" cargo test test_no_cpu_fallback  # MUST FAIL
```

### Weeks 3-4: Parallel Sampling Implementation
Files to create IN ORDER:

**Week 3: Core Architecture**
1. `crates/prism-validation/src/pdb_sanitizer.rs`
2. `crates/prism-validation/src/sampling/mod.rs`
3. `crates/prism-validation/src/sampling/contract.rs` (THE LAW)
4. `crates/prism-validation/src/sampling/result.rs`
5. `crates/prism-validation/src/sampling/paths/mod.rs`
6. `crates/prism-validation/src/sampling/paths/nova_path.rs` (GREENFIELD)
7. `crates/prism-validation/src/sampling/paths/amber_path.rs` (STABLE)
8. `crates/prism-validation/src/sampling/router/mod.rs`

**Week 4: Shadow Pipeline + Migration**
9. `crates/prism-validation/src/sampling/shadow/mod.rs`
10. `crates/prism-validation/src/sampling/shadow/comparator.rs`
11. `crates/prism-validation/src/sampling/migration/mod.rs`
12. `crates/prism-validation/src/sampling/migration/feature_flags.rs`
13. `crates/prism-validation/src/apo_holo_benchmark.rs`

ISOLATION CHECK (CRITICAL):
```bash
# Verify nova_path.rs does NOT import from amber_path.rs
grep -r "amber_path" crates/prism-validation/src/sampling/paths/nova_path.rs
# Expected: No output (empty)

# Verify amber_path.rs does NOT import from nova_path.rs
grep -r "nova_path" crates/prism-validation/src/sampling/paths/amber_path.rs
# Expected: No output (empty)
```

CONTRACT COMPLIANCE CHECK:
```bash
# Both paths must implement SamplingBackend
cargo test --release -p prism-validation --features hybrid-sampler contract_tests
```

CHECKPOINT: Must pass before proceeding:
```bash
cargo run --release -p prism-validation --bin apo-holo-single -- --apo 1AKE --holo 4AKE
# Expected: min RMSD < 3.5A

# Shadow comparison must show Equivalent or MinorDivergence
cargo run --release -p prism-validation --bin shadow-compare -- --pdb test.pdb
```

### Weeks 5-6: CryptoBench & Ablation
Files to create IN ORDER:
1. `crates/prism-validation/src/cryptobench_dataset.rs`
2. `crates/prism-validation/src/ablation.rs`
3. `crates/prism-validation/src/failure_analysis.rs`

CHECKPOINT: Must pass before proceeding:
```bash
cargo run --release -p prism-validation --bin cryptobench -- --manifest manifest.json
# Expected: ROC AUC > 0.70
```

### Weeks 7-8: Publication
Files to create:
1. `crates/prism-validation/src/publication_outputs.rs`
2. `scripts/generate_figures.py`

## Verification Requirements

Before marking ANY task complete, Claude MUST:

1. **Compile check**: `cargo check -p prism-validation --features cuda`
2. **Run relevant tests**: `cargo test --release -p prism-validation --features cuda [module]`
3. **Verify no regression**: Compare metrics to baseline
4. **Log checkpoint**: Report actual numeric results

## Code Quality Standards

Every new file MUST include:
- Module-level documentation with `//!` comments
- Zero Fallback Policy statement in GPU code
- Unit tests for core functionality
- Error handling with `anyhow::Result`
- No `unwrap()` in production paths (use `context()`)

## Commit Requirements

All commits must:
- Reference the Phase 6 plan
- Include test results in commit message if applicable
- Use conventional commit format: `feat(validation):`, `fix(gpu):`, etc.

## When Stuck or Uncertain

1. Re-read `results/phase6_sota_plan.md` section for current task
2. Check existing code patterns in `crates/prism-validation/src/`
3. Ask user for clarification rather than guessing
4. Never skip a checkpoint - if tests fail, fix before proceeding

## Success Metrics (Non-Negotiable)

| Metric | Minimum | Target |
|--------|---------|--------|
| ROC AUC | 0.70 | >0.75 |
| PR AUC | 0.20 | >0.25 |
| Success Rate | 80% | >85% |
| Ablation Delta | +0.15 | >+0.20 |
| Apo-Holo Success | 60% | >66% |

## Phase Sizing and Context Management

### CRITICAL: Implementation Must Be Phased

To maintain context and alignment, Claude MUST:

1. **Implement ONE file at a time** - Complete, test, and commit each file before starting the next
2. **Run checkpoint after each file** - Verify compilation and basic tests pass
3. **Commit after each completed file** - Creates restore points
4. **Re-read the plan section** - Before starting each new file

### Maximum Scope Per Session

| Action | Maximum Scope | Verification Required |
|--------|---------------|----------------------|
| New file creation | 1 file | Compiles, tests pass |
| File modification | 2-3 functions | Tests pass |
| Bug fix | Single issue | Regression test |
| Integration | 1 module connection | Integration test |

### Mandatory Context Refresh

Before starting ANY new file, Claude MUST:
1. Read the relevant section of `results/phase6_sota_plan.md`
2. Read existing related files in `crates/prism-validation/src/`
3. State the specific code from the plan being implemented

### Session Boundaries

Each session should follow this structure:
```
1. Read CLAUDE.md (automatic)
2. Check current phase: ./scripts/phase6_checkpoint.sh auto
3. Identify next file to implement
4. Re-read plan section for that file
5. Implement the file
6. Run tests
7. Commit
8. Verify checkpoint still passes
9. STOP - Allow user verification before next file
```

### Context Loss Prevention

If Claude shows signs of context drift:
- Implementing features not in the plan
- Using wrong parameter values
- Skipping required tests
- Not following the file order

User should:
1. Stop Claude immediately
2. Run `./scripts/phase6_compliance_check.sh`
3. Point Claude to specific plan section
4. Resume with explicit file target

### Atomic Implementation Units

Each implementation unit must be self-contained:

**Week 1-2 Units:**
```
Unit 1.1: cryptic_features.rs (complete struct + tests)
  -> Commit -> Checkpoint

Unit 1.2: gpu_zro_cryptic_scorer.rs (scorer struct + new())
  -> Commit -> Checkpoint

Unit 1.3: gpu_zro_cryptic_scorer.rs (score_residue + score_and_learn)
  -> Commit -> Checkpoint

Unit 1.4: gpu_zro_cryptic_scorer.rs (RLS update + stability)
  -> Commit -> Checkpoint

Unit 1.5: tests/gpu_scorer_tests.rs (all tests)
  -> Commit -> Full Week 2 Checkpoint
```

**Week 3-4 Units:**
```
Unit 3.1: pdb_sanitizer.rs
  -> Commit -> Checkpoint

Unit 3.2: nova_cryptic_sampler.rs (config + new)
  -> Commit -> Checkpoint

Unit 3.3: nova_cryptic_sampler.rs (sample method)
  -> Commit -> Checkpoint

Unit 3.4: apo_holo_benchmark.rs (structs + pairs)
  -> Commit -> Checkpoint

Unit 3.5: apo_holo_benchmark.rs (run methods)
  -> Commit -> Full Week 4 Checkpoint
```

**Week 5-6 Units:**
```
Unit 5.1: cryptobench_dataset.rs
  -> Commit -> Checkpoint

Unit 5.2: ablation.rs (variants + results)
  -> Commit -> Checkpoint

Unit 5.3: ablation.rs (study runner)
  -> Commit -> Checkpoint

Unit 5.4: failure_analysis.rs
  -> Commit -> Full Week 6 Checkpoint
```

**Week 7-8 Units:**
```
Unit 7.1: publication_outputs.rs (structs)
  -> Commit -> Checkpoint

Unit 7.2: publication_outputs.rs (table generators)
  -> Commit -> Checkpoint

Unit 7.3: generate_figures.py
  -> Commit -> Full Week 8 Checkpoint
```

### Session Start Protocol

At the START of every implementation session, Claude MUST:

```
1. State: "Beginning Phase 6 implementation session"
2. Report: Current checkpoint status
3. Identify: Next implementation unit
4. Quote: Relevant plan section
5. Confirm: "Implementing [file] - [unit description]"
6. Request: User confirmation to proceed
```

### Session End Protocol

At the END of every session (or after each unit), Claude MUST:

```
1. Report: What was implemented
2. Show: Test results
3. Commit: With descriptive message
4. Run: ./scripts/phase6_checkpoint.sh auto
5. State: "Ready for next unit: [description]"
6. STOP: Wait for user approval
```

## File Locations

- Phase 6 Plan: `results/phase6_sota_plan.md`
- Phase 7-8 Plan: `results/phase7_8_sota_plan.md`
- Implementation: `crates/prism-validation/src/`
- Tests: `crates/prism-validation/src/tests/`
- Data: `data/benchmarks/`
- Results: `results/`
- Compliance: `scripts/phase6_*.sh`

## Obsidian Vault (REQUIRED for Progress Tracking)

Location: `.obsidian/vault/`

**Claude MUST read and update these files:**

| File | Purpose | When to Update |
|------|---------|----------------|
| `Sessions/Current Session.md` | Active work context | Every significant action |
| `Progress/implementation_status.json` | Machine-readable status | After each file completion |
| `PRISM Dashboard.md` | Human-readable overview | End of session |

### Session Start Protocol (Updated)

At the START of every session, Claude MUST:

```
1. Read .obsidian/vault/Sessions/Current Session.md
2. Read .obsidian/vault/Progress/implementation_status.json
3. Parse next_action from status JSON
4. State current target and context
5. Request confirmation to proceed
```

### Session End Protocol (Updated)

At the END of every session, Claude MUST:

```
1. Update .obsidian/vault/Sessions/Current Session.md with:
   - Work completed
   - Context for continuation
   - Next target
   - Any blocking issues

2. Update .obsidian/vault/Progress/implementation_status.json with:
   - File completion status
   - Checkpoint progress
   - Metrics if available
   - next_action field

3. Update .obsidian/vault/PRISM Dashboard.md with:
   - Overall progress percentage
   - Recent session summary
   - Next action

4. Commit vault changes:
   git add .obsidian/vault/ && git commit -m "vault: Update progress"
```

### Vault Update Commands

```bash
# After completing a file
Claude should update implementation_status.json:
- Set file status to "completed"
- Set completed_at to current timestamp
- Set commit to the commit hash
- Update next_action to next file

# After passing a checkpoint
Claude should update:
- checkpoints.week_X.status to "passed"
- checkpoints.week_X.passed_at to timestamp
- current_week to next week number
```

## Future: Phase 7-8 (After Phase 6 Complete)

**DO NOT IMPLEMENT UNTIL PHASE 6 GATES PASS**

Reference: `results/phase7_8_sota_plan.md`

Phase 7-8 builds on the AMBER-primary foundation with:
- Hierarchical reservoir (1,280 neurons)
- Full persistence diagrams (extends Betti numbers)
- Ensemble voting (5 reservoirs)
- Transfer learning (family backbones)
- Uncertainty quantification

Target: ROC AUC 0.90 (exceeds SOTA 0.87)

**Gate Requirements to begin Phase 7:**
```
[ ] Phase 6 ROC AUC >= 0.70
[ ] All Phase 6 tests passing
[ ] results/PHASE6_FINAL.json exists
[ ] User approval to proceed
```
