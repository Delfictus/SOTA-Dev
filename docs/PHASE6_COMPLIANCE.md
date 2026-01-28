# Phase 6 Compliance Framework

This document describes the mechanisms in place to ensure Claude fully complies with the Phase 6 SOTA Implementation Plan.

## Overview

The compliance framework consists of 5 layers:

1. **CLAUDE.md** - Session-level instructions read at startup
2. **Checkpoint Gates** - Must-pass tests before proceeding
3. **Compliance Checker** - Verifies code follows specifications
4. **Pre-commit Hooks** - Blocks non-compliant commits
5. **Locked Specifications** - Prevents plan modification

## How to Use

### Before Releasing Claude to Work

1. Ensure `CLAUDE.md` exists in project root
2. Run initial compliance check:
   ```bash
   ./scripts/phase6_compliance_check.sh
   ```
3. Verify Week 0 checkpoint:
   ```bash
   ./scripts/phase6_checkpoint.sh 0
   ```

### During Implementation

After each major implementation phase, run the checkpoint:

```bash
# After Weeks 1-2 (GPU SNN)
./scripts/phase6_checkpoint.sh 2

# After Weeks 3-4 (NOVA)
./scripts/phase6_checkpoint.sh 4

# After Weeks 5-6 (CryptoBench)
./scripts/phase6_checkpoint.sh 6

# Final validation
./scripts/phase6_checkpoint.sh 8
```

### Periodic Compliance Verification

Run the full compliance check at any time:

```bash
./scripts/phase6_compliance_check.sh
```

## Compliance Mechanisms Explained

### 1. CLAUDE.md (Project Instructions)

Location: `/CLAUDE.md`

Claude Code reads this file at the start of every session. It contains:
- Non-negotiable constraints (zero fallback, zero external deps)
- Exact file creation order
- Checkpoint requirements before proceeding
- Success metrics

**Key enforcements:**
- Explicit fail on missing GPU (no silent fallback)
- 512-neuron reservoir size
- RLS lambda = 0.99
- 16-dim feature vector with velocity encoding

### 2. Checkpoint Gates

Location: `/scripts/phase6_checkpoint.sh`

These are hard gates that MUST pass before proceeding:

| Week | Gate | Critical Test |
|------|------|---------------|
| 0 | Environment | Rust 1.75+, CUDA present |
| 2 | GPU SNN | Zero fallback test FAILS without GPU |
| 4 | NOVA | Apo-holo RMSD < 3.5A |
| 6 | CryptoBench | ROC AUC > 0.70 |
| 8 | Final | All metrics meet minimums |

### 3. Compliance Checker

Location: `/scripts/phase6_compliance_check.sh`

Verifies:
- No fallback patterns in GPU code
- All required files present
- Module documentation exists
- No todo!() in production code
- 16-dim feature vector
- No forbidden dependencies
- Dataset integrity

### 4. Pre-commit Hooks

The existing pre-commit hook already blocks:
- unwrap() abuse
- Hardcoded values
- Mock implementations

Phase 6 adds checking for:
- Zero fallback violations
- Missing error handling

### 5. Locked Specifications

These files should NOT be modified without explicit approval:

| File | Reason |
|------|--------|
| `results/phase6_sota_plan.md` | Authoritative specification |
| `CLAUDE.md` | Compliance instructions |
| `scripts/phase6_*.sh` | Verification scripts |

## What Claude Cannot Do

Per the compliance framework, Claude is FORBIDDEN from:

1. **Falling back to CPU silently** - Must throw explicit error
2. **Using external ML frameworks** - PyTorch, TensorFlow, etc.
3. **Skipping checkpoint tests** - Must pass before next phase
4. **Using mock/placeholder data** - All values must be computed
5. **Modifying the plan** - Without explicit user approval
6. **Using todo!() or unimplemented!()** - In production code
7. **Leaking test data** - Into training

## Verification Commands Summary

```bash
# Full compliance check
./scripts/phase6_compliance_check.sh

# Week-specific checkpoints
./scripts/phase6_checkpoint.sh 0   # Environment
./scripts/phase6_checkpoint.sh 2   # GPU SNN
./scripts/phase6_checkpoint.sh 4   # NOVA
./scripts/phase6_checkpoint.sh 6   # CryptoBench
./scripts/phase6_checkpoint.sh 8   # Final

# Manual critical test
CUDA_VISIBLE_DEVICES="" cargo test test_no_cpu_fallback
# This MUST FAIL - if it passes, there's a hidden fallback

# Throughput check
cargo test bench_gpu_scorer_throughput -- --nocapture
# Should show >10,000 residues/second
```

## Trust But Verify

While this framework provides strong guardrails, you should:

1. **Run checkpoints** after each major phase
2. **Review commits** for compliance with the plan
3. **Check metrics** against the targets in the plan
4. **Verify the zero fallback test fails** without GPU

## Troubleshooting

### Checkpoint Fails

1. Read the specific failure message
2. Check `results/phase6_sota_plan.md` for requirements
3. Fix the issue before allowing Claude to proceed

### Compliance Check Fails

1. Run `./scripts/phase6_compliance_check.sh` for details
2. Address each FAIL item
3. Re-run until all pass

### Unexpected Behavior

1. Verify `CLAUDE.md` hasn't been modified
2. Check that checkpoint was run for current phase
3. Review recent commits against the plan
