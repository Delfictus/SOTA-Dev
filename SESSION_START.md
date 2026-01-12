# Starting a New Claude Code Session for PRISM Phase 6

## Quick Start

### Option 1: Run Initialization Script (Recommended)

```bash
cd /home/diddy/Desktop/PRISM4D-bio
./scripts/init_phase6_session.sh
```

Then copy the provided prompt into Claude Code.

### Option 2: Manual Initialization

Start Claude Code and paste this prompt:

```
Begin Phase 6 implementation session.

1. Read CLAUDE.md for project instructions
2. Read results/phase6_sota_plan.md for implementation plan
3. Run: ./scripts/phase6_checkpoint.sh auto
4. Identify and implement the next file in order
5. Follow atomic implementation units (one file at a time)

What is the current checkpoint status and next implementation target?
```

---

## What Claude Will Do

When properly initialized, Claude will:

1. **Read CLAUDE.md** - Contains all constraints, architecture, and implementation order
2. **Read the Phase 6 plan** - Full specifications for each file
3. **Check current progress** - Determine which files are done
4. **Identify next target** - Pick the next file in the required order
5. **Request confirmation** - Ask before implementing

---

## Implementation Order (Phase 6)

Claude MUST implement files in this exact order:

### Week 0: Setup
- Environment verification
- Dataset download

### Weeks 1-2: Core Scoring
1. `cryptic_features.rs`
2. `gpu_zro_cryptic_scorer.rs`
3. `tests/gpu_scorer_tests.rs`

### Weeks 3-4: Parallel Sampling
4. `pdb_sanitizer.rs`
5. `sampling/contract.rs`
6. `sampling/paths/nova_path.rs`
7. `sampling/paths/amber_path.rs`
8. `sampling/router/mod.rs`
9. `sampling/shadow/comparator.rs`
10. `sampling/migration/feature_flags.rs`
11. `apo_holo_benchmark.rs`

### Weeks 5-6: Benchmarking
12. `cryptobench_dataset.rs`
13. `ablation.rs`
14. `failure_analysis.rs`

### Weeks 7-8: Publication
15. `publication_outputs.rs`

---

## Verification Commands

After each file, Claude should run:

```bash
# Compile check
cargo check -p prism-validation --features cuda

# Run tests for the module
cargo test --release -p prism-validation --features cuda [module_name]

# Check compliance
./scripts/phase6_compliance_check.sh
```

---

## If Claude Loses Context

Signs of context drift:
- Implementing files out of order
- Using wrong parameter values (neurons != 512, lambda != 0.99)
- Skipping tests
- Not following AMBER-primary architecture

Recovery steps:
1. Stop Claude
2. Run `./scripts/phase6_compliance_check.sh`
3. Restart with the initialization prompt above
4. Point Claude to the specific plan section needed

---

## Key Documents

| Document | Purpose |
|----------|---------|
| `CLAUDE.md` | Session instructions, constraints, order |
| `results/phase6_sota_plan.md` | Full Phase 6 specifications |
| `results/phase7_8_sota_plan.md` | Future phases (don't implement yet) |
| `scripts/phase6_checkpoint.sh` | Gate verification |
| `scripts/phase6_compliance_check.sh` | Code quality checks |

---

## Architecture Reminders

```
AMBER-PRIMARY:
- AMBER is the main physics engine (all structures)
- NOVA optional for small proteins (≤512 atoms)
- Parallel paths with shadow pipeline

CORE FEATURES:
- Betti numbers (beta_0, beta_1, beta_2) for TDA
- Blake3 hashing for integrity/caching
- RLS online learning (lambda=0.99)
- 512-neuron reservoir

ZERO POLICIES:
- Zero CPU fallback (explicit GPU error)
- Zero external ML dependencies
- Zero mock implementations
- Zero data leakage
```

---

## Success Metrics

Phase 6 is complete when:

| Metric | Minimum |
|--------|---------|
| ROC AUC | ≥ 0.70 |
| PR AUC | ≥ 0.20 |
| Success Rate | ≥ 80% |
| All tests | Passing |
