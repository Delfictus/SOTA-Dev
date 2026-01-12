# PRISM Master Implementation Trajectory
## Phase 6 → Phase 7 → Phase 8: Unified Roadmap

**Document Version**: 1.0  
**Created**: 2026-01-12  
**Classification**: Master Planning Document  
**Status**: Phase 6 Ready for Execution | Phase 7-8 Documented for Future  

---

## Document Purpose

This document establishes the **complete implementation trajectory** for PRISM cryptic site detection, ensuring:

1. **Phase 6** remains intact and ready for immediate execution
2. **Phase 7-8** enhancements are documented and aligned with Phase 6 outputs
3. **Parallel implementation architecture** is planned from the start
4. **No breaking changes** occur when transitioning between phases

**CRITICAL**: Phase 6 must be **fully completed and validated** before Phase 7 begins.

---

## Executive Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PRISM IMPLEMENTATION TRAJECTORY                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PHASE 6: Foundation (8 weeks)                     Target: 0.75 AUC        │
│  ══════════════════════════════                                            │
│  • GPU ZrO Scorer (512 neurons)                                            │
│  • NOVA Cryptic Sampler (500 conformations)                                │
│  • CryptoBench Validation                                                  │
│  • Hybrid NOVA/AMBER Router (parallel paths)                               │
│  STATUS: ██████████ READY FOR EXECUTION                                    │
│                                                                             │
│         ↓ (Checkpoint: AUC ≥ 0.70 required to proceed)                     │
│                                                                             │
│  PHASE 7: Architecture Enhancement (8 weeks)       Target: 0.82 AUC        │
│  ═══════════════════════════════════════════                               │
│  • Hierarchical Reservoir (1,280 neurons)                                  │
│  • Persistent Homology (full TDA)                                          │
│  • Extended Sampling (2,000 conformations)                                 │
│  • Multi-Scale Features (67 dimensions)                                    │
│  STATUS: ░░░░░░░░░░ DOCUMENTED, AWAITING PHASE 6                           │
│                                                                             │
│         ↓ (Checkpoint: AUC ≥ 0.80 required to proceed)                     │
│                                                                             │
│  PHASE 8: Advanced Capabilities (8 weeks)          Target: 0.90 AUC        │
│  ════════════════════════════════════════                                  │
│  • Ensemble Voting (5 reservoirs)                                          │
│  • Transfer Learning (family backbones)                                    │
│  • Uncertainty Quantification                                              │
│  • Active Learning Pipeline                                                │
│  STATUS: ░░░░░░░░░░ DOCUMENTED, AWAITING PHASE 7                           │
│                                                                             │
│  TOTAL TIMELINE: 24 weeks                                                  │
│  FINAL TARGET: 0.90 AUC (exceeds SOTA 0.87)                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase Transition Requirements

### Phase 6 → Phase 7 Gate

**MANDATORY REQUIREMENTS** (all must pass):

```
□ CryptoBench ROC AUC ≥ 0.70
□ CryptoBench PR AUC ≥ 0.20
□ Success Rate ≥ 80% on apo-holo benchmark
□ GPU scorer operational (512 neurons, <5ms/residue)
□ NOVA sampling functional (500 conformations)
□ Hybrid router tested (NOVA + AMBER paths)
□ All Phase 6 tests passing
□ Zero external dependencies verified
□ Results documented in results/PHASE6_FINAL.json
```

**DO NOT PROCEED TO PHASE 7 UNTIL ALL BOXES CHECKED**

### Phase 7 → Phase 8 Gate

**MANDATORY REQUIREMENTS**:

```
□ CryptoBench ROC AUC ≥ 0.80
□ CryptoBench PR AUC ≥ 0.30
□ Hierarchical reservoir operational (1,280 neurons)
□ Persistent homology features extracted
□ Extended sampling working (2,000 conformations)
□ All Phase 7 tests passing
□ Shadow pipeline validation passed (vs Phase 6)
□ Results documented in results/PHASE7_FINAL.json
```

---

## Architectural Continuity

### Parallel Implementation Pattern

From Phase 6 onward, PRISM uses a **parallel implementation with shadow pipeline** architecture:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PARALLEL IMPLEMENTATION ARCHITECTURE                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                         ┌─────────────────┐                                │
│                         │  ROUTER LAYER   │                                │
│                         │ (SamplingRouter)│                                │
│                         └────────┬────────┘                                │
│                                  │                                          │
│              ┌───────────────────┼───────────────────┐                     │
│              │                   │                   │                     │
│              ▼                   ▼                   ▼                     │
│   ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐          │
│   │    NOVA PATH     │ │   AMBER PATH     │ │   SHADOW PATH    │          │
│   │   (Greenfield)   │ │    (Stable)      │ │  (Comparison)    │          │
│   │                  │ │                  │ │                  │          │
│   │ Phase 6: Base    │ │ Phase 6: Base    │ │ Runs both paths  │          │
│   │ Phase 7: Enhanced│ │ Phase 7: Stable  │ │ Compares outputs │          │
│   │ Phase 8: Advanced│ │ Phase 8: Stable  │ │ Validates before │          │
│   │                  │ │                  │ │ promotion        │          │
│   └──────────────────┘ └──────────────────┘ └──────────────────┘          │
│              │                   │                   │                     │
│              └───────────────────┼───────────────────┘                     │
│                                  ▼                                          │
│                    ┌─────────────────────────┐                             │
│                    │   UNIFIED OUTPUT        │                             │
│                    │   (SamplingResult)      │                             │
│                    └─────────────────────────┘                             │
│                                                                             │
│  KEY PRINCIPLE: NOVA path evolves through phases while AMBER remains       │
│  stable. Shadow path validates greenfield before promotion.                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Contract Stability

**The `SamplingBackend` trait is THE CONTRACT.** All phases must satisfy it:

```rust
/// Contract Version: 1.0.0
/// STABLE ACROSS ALL PHASES
pub trait SamplingBackend: Send + Sync {
    fn id(&self) -> BackendId;
    fn capabilities(&self) -> BackendCapabilities;
    fn load_structure(&mut self, structure: &SanitizedStructure) -> Result<()>;
    fn sample(&mut self, config: &SamplingConfig) -> Result<SamplingResult>;
    fn reset(&mut self) -> Result<()>;
    fn estimate_vram_mb(&self, n_atoms: usize) -> f32;
}
```

**Phase 7-8 enhancements extend capabilities WITHOUT breaking the contract.**

---

## Implementation Order

### CURRENT: Phase 6 (Execute Now)

**Reference Documents:**
- `results/phase6_sota_plan.md` (or consolidated equivalent)
- `PRISM_PHASE6_QUICK_REFERENCE.md`

**Implementation Files (in order):**

```
Week 0: Setup
├── Download CryptoBench dataset
├── Verify CUDA environment
└── Create test fixtures

Week 1-2: Core Components
├── pdb_sanitizer.rs
├── cryptic_features.rs (16-dim)
├── gpu_zro_scorer.rs (512 neurons)
└── tests/

Week 3-4: Sampling Infrastructure
├── sampling/contract.rs (THE CONTRACT)
├── sampling/paths/nova_path.rs
├── sampling/paths/amber_path.rs
├── sampling/router/mod.rs (hybrid router)
└── nova_cryptic_sampler.rs

Week 5-6: Benchmarking
├── cryptobench_loader.rs
├── apo_holo_benchmark.rs
├── metrics.rs (ROC AUC, PR AUC)
└── ablation.rs

Week 7-8: Validation + Documentation
├── Run full CryptoBench evaluation
├── Generate results/PHASE6_FINAL.json
├── Verify all gate requirements
└── Document lessons learned
```

**Success Criteria:**
```
□ ROC AUC ≥ 0.70 (target 0.75)
□ PR AUC ≥ 0.20 (target 0.25)
□ Success Rate ≥ 80%
□ <1s per structure
□ Zero external dependencies
```

---

### FUTURE: Phase 7 (After Phase 6 Complete)

**Reference Documents:**
- `PRISM_PHASE7_8_PLAN_PART1.md`
- `PRISM_PHASE7_8_QUICK_REFERENCE.md`

**Prerequisites:**
- [ ] Phase 6 gate requirements passed
- [ ] `results/PHASE6_FINAL.json` exists with AUC ≥ 0.70

**Implementation Files (in order):**

```
Week 1-2: Hierarchical Reservoir
├── hierarchical_reservoir.rs (1,280 neurons)
├── kernels/hierarchical_reservoir.cu
└── tests/hierarchical_tests.rs

Week 3-4: TDA + Features
├── persistent_homology.rs (31-dim)
├── multiscale_features.rs (67-dim total)
├── kernels/persistence.cu
└── tests/tda_tests.rs

Week 5-6: Extended Sampling
├── extended_nova_sampler.rs (2,000 samples)
├── Adaptive biasing implementation
├── Temperature annealing
└── tests/sampling_tests.rs

Week 7-8: Integration + Validation
├── phase7_scorer.rs (integrated pipeline)
├── Shadow validation vs Phase 6
├── Run CryptoBench evaluation
└── Generate results/PHASE7_FINAL.json
```

**Success Criteria:**
```
□ ROC AUC ≥ 0.80 (target 0.82)
□ PR AUC ≥ 0.30 (target 0.32)
□ Shadow comparison: no critical divergence from Phase 6
□ <2s per structure
□ Zero external dependencies maintained
```

---

### FUTURE: Phase 8 (After Phase 7 Complete)

**Reference Documents:**
- `PRISM_PHASE7_8_PLAN_PART2.md`
- `PRISM_PHASE7_8_QUICK_REFERENCE.md`

**Prerequisites:**
- [ ] Phase 7 gate requirements passed
- [ ] `results/PHASE7_FINAL.json` exists with AUC ≥ 0.80

**Implementation Files (in order):**

```
Week 9-10: Ensemble Voting
├── ensemble_reservoir.rs (5 reservoirs)
├── Learned combination weights
└── tests/ensemble_tests.rs

Week 11-13: Transfer Learning
├── transfer_learning.rs
├── Family backbone extraction
├── Cross-structure transfer
└── tests/transfer_tests.rs

Week 14: Uncertainty
├── uncertainty.rs
├── Calibration pipeline
└── tests/calibration_tests.rs

Week 15-16: Active Learning + Final
├── active_learning.rs
├── Structure prioritization
├── phase8_scorer.rs (complete system)
├── Run final CryptoBench evaluation
└── Generate results/PHASE8_FINAL.json
```

**Success Criteria:**
```
□ ROC AUC ≥ 0.88 (target 0.90)
□ PR AUC ≥ 0.38 (target 0.40)
□ Exceeds or matches PocketMiner (0.87 AUC)
□ Uncertainty ECE < 0.10
□ <3s per structure (with ensemble)
□ Zero external dependencies maintained
□ Publication-ready results
```

---

## File Organization

### Directory Structure (All Phases)

```
crates/prism-validation/src/
├── lib.rs
│
├── sampling/                          # PARALLEL IMPLEMENTATION
│   ├── mod.rs                         # Public API
│   ├── contract.rs                    # SamplingBackend trait (STABLE)
│   ├── result.rs                      # SamplingResult types (STABLE)
│   │
│   ├── paths/                         # ISOLATED BACKENDS
│   │   ├── mod.rs
│   │   ├── nova_path.rs               # Phase 6: Base → Phase 7-8: Enhanced
│   │   ├── amber_path.rs              # All phases: Stable reference
│   │   └── mock_path.rs               # Testing
│   │
│   ├── router/                        # ROUTING LAYER
│   │   ├── mod.rs
│   │   ├── auto_router.rs             # Size-based routing
│   │   ├── shadow_runner.rs           # Shadow comparison
│   │   └── strategy.rs                # Routing strategies
│   │
│   ├── shadow/                        # SHADOW PIPELINE
│   │   ├── mod.rs
│   │   ├── comparator.rs              # Output comparison
│   │   └── divergence_log.rs          # Difference tracking
│   │
│   └── migration/                     # STRANGLER PATTERN
│       ├── mod.rs
│       ├── feature_flags.rs           # Gradual rollout
│       └── rollback.rs                # Automatic rollback
│
├── scoring/                           # NEUROMORPHIC SCORING
│   ├── mod.rs
│   ├── gpu_zro_scorer.rs              # Phase 6: 512 neurons
│   ├── hierarchical_reservoir.rs      # Phase 7: 1,280 neurons
│   └── ensemble_reservoir.rs          # Phase 8: 5 × 1,280 neurons
│
├── features/                          # FEATURE EXTRACTION
│   ├── mod.rs
│   ├── cryptic_features.rs            # Phase 6: 16-dim
│   ├── multiscale_features.rs         # Phase 7: 36-dim
│   └── persistent_homology.rs         # Phase 7: +31-dim
│
├── benchmarks/                        # VALIDATION
│   ├── mod.rs
│   ├── cryptobench_loader.rs          # Dataset loading
│   ├── apo_holo_benchmark.rs          # Apo-holo pairs
│   └── metrics.rs                     # ROC AUC, PR AUC
│
├── transfer/                          # PHASE 8: TRANSFER LEARNING
│   ├── mod.rs
│   ├── transfer_learning.rs
│   └── family_backbone.rs
│
├── uncertainty/                       # PHASE 8: UNCERTAINTY
│   ├── mod.rs
│   ├── uncertainty.rs
│   └── calibration.rs
│
└── active/                            # PHASE 8: ACTIVE LEARNING
    ├── mod.rs
    └── active_learning.rs

results/
├── PHASE6_FINAL.json                  # Phase 6 completion record
├── PHASE7_FINAL.json                  # Phase 7 completion record
├── PHASE8_FINAL.json                  # Phase 8 completion record
└── shadow_comparisons/                # Shadow pipeline logs
```

---

## Version Compatibility Matrix

| Component | Phase 6 | Phase 7 | Phase 8 |
|-----------|---------|---------|---------|
| `SamplingBackend` trait | v1.0.0 | v1.0.0 | v1.0.0 |
| `SamplingResult` struct | v1.0.0 | v1.0.0 | v1.0.0 |
| `CrypticFeatures` | 16-dim | 67-dim (backward compat) | 67-dim |
| `gpu_zro_scorer` | 512 neurons | Deprecated (use hierarchical) | Deprecated |
| `hierarchical_reservoir` | - | 1,280 neurons | 1,280 neurons |
| `ensemble_reservoir` | - | - | 5 × 1,280 neurons |
| `NovaPath` | Base TDA | Extended TDA | Extended TDA |
| `AmberPath` | Stable | Stable | Stable |

**Backward Compatibility Rules:**
1. `SamplingResult` format NEVER changes (add optional fields only)
2. `SamplingBackend` trait NEVER changes (add extension traits)
3. Phase N code can process Phase N-1 outputs
4. Shadow comparison validates compatibility

---

## Claude Code Instructions

### When Starting Phase 6

```
INSTRUCTION: Execute Phase 6 implementation as documented in:
- results/phase6_sota_plan.md (primary)
- PRISM_PHASE6_QUICK_REFERENCE.md (reference)

DO:
- Implement all Phase 6 components in order
- Create parallel NOVA/AMBER paths from start
- Establish SamplingBackend contract
- Validate against CryptoBench

DO NOT:
- Skip to Phase 7-8 components
- Modify contract.rs after initial creation
- Break AMBER path while developing NOVA
- Proceed past Phase 6 gate without approval

AWARENESS:
- Phase 7-8 plans exist in PRISM_PHASE7_8_*.md
- Current architecture supports future enhancements
- Shadow pipeline will validate Phase 7 against Phase 6
```

### When Starting Phase 7 (Future)

```
INSTRUCTION: Execute Phase 7 implementation as documented in:
- PRISM_PHASE7_8_PLAN_PART1.md (primary)
- PRISM_PHASE7_8_QUICK_REFERENCE.md (reference)

PREREQUISITES:
- Verify results/PHASE6_FINAL.json exists
- Verify ROC AUC ≥ 0.70 in Phase 6 results
- All Phase 6 tests passing

DO:
- Implement hierarchical reservoir as ENHANCEMENT to nova_path.rs
- Add persistent homology features (maintain backward compat)
- Run shadow comparison against Phase 6 results
- Validate metrics improvement

DO NOT:
- Modify SamplingBackend contract
- Break Phase 6 functionality
- Remove 512-neuron scorer (keep for fallback)
- Proceed without shadow validation
```

### When Starting Phase 8 (Future)

```
INSTRUCTION: Execute Phase 8 implementation as documented in:
- PRISM_PHASE7_8_PLAN_PART2.md (primary)
- PRISM_PHASE7_8_QUICK_REFERENCE.md (reference)

PREREQUISITES:
- Verify results/PHASE7_FINAL.json exists
- Verify ROC AUC ≥ 0.80 in Phase 7 results
- Shadow comparison shows no critical divergence

DO:
- Implement ensemble as wrapper around hierarchical reservoir
- Add transfer learning infrastructure
- Implement uncertainty quantification
- Run final validation against all benchmarks

DO NOT:
- Modify contract (add extension traits if needed)
- Remove Phase 7 components (build on top)
- Skip uncertainty calibration
- Claim SOTA without verification
```

---

## Risk Mitigation

### Phase Transition Risks

| Risk | Mitigation |
|------|------------|
| Phase 6 fails to meet AUC target | Iterate on Phase 6 before proceeding |
| Phase 7 breaks Phase 6 functionality | Shadow pipeline catches divergence |
| Enhancement path diverges too far | Strangler pattern allows rollback |
| Memory/VRAM constraints | Test on RTX 3060 at each phase |
| External dependency creep | Verify at each gate checkpoint |

### Rollback Procedures

**Phase 7 → Phase 6 Rollback:**
```bash
# If Phase 7 causes issues
git checkout phase6-final
cargo test --release -p prism-validation
# Verify Phase 6 still works
```

**Phase 8 → Phase 7 Rollback:**
```bash
# If Phase 8 causes issues
git checkout phase7-final
cargo test --release -p prism-validation
# Verify Phase 7 still works
```

**Feature Flag Rollback:**
```rust
// In router, can force stable path
let router = SamplingRouter::new(context)?
    .with_migration_stage(MigrationStage::StableOnly);
```

---

## Document References

### Phase 6 (Current)

| Document | Purpose |
|----------|---------|
| `results/phase6_sota_plan.md` | Primary implementation plan |
| `PRISM_PHASE6_QUICK_REFERENCE.md` | Task checklist and metrics |
| This document | Trajectory alignment |

### Phase 7-8 (Future)

| Document | Purpose |
|----------|---------|
| `PRISM_PHASE7_8_PLAN_PART1.md` | Weeks 1-4: Hierarchical + TDA |
| `PRISM_PHASE7_8_PLAN_PART2.md` | Weeks 5-16: Ensemble + Transfer |
| `PRISM_PHASE7_8_QUICK_REFERENCE.md` | Summary and checkpoints |
| This document | Trajectory alignment |

---

## Summary

**CURRENT ACTION**: Execute Phase 6 as documented. The parallel implementation architecture and shadow pipeline are part of Phase 6 to support future phases.

**FUTURE AWARENESS**: Phase 7-8 enhancement plans exist and are designed to build on Phase 6 outputs without breaking changes. Do not implement Phase 7-8 components until Phase 6 gate requirements are met.

**ALIGNMENT GUARANTEE**: Following this trajectory ensures:
1. Each phase builds correctly on the previous
2. No breaking changes between phases
3. Shadow validation catches regressions
4. Rollback is always possible
5. Sovereignty is maintained throughout

---

## Approval Checkpoints

### Phase 6 Completion
```
Date: ____________
ROC AUC Achieved: ____________
All Gates Passed: □ Yes □ No
Approved for Phase 7: □ Yes □ No
Signature: ____________
```

### Phase 7 Completion
```
Date: ____________
ROC AUC Achieved: ____________
Shadow Validation: □ Pass □ Fail
Approved for Phase 8: □ Yes □ No
Signature: ____________
```

### Phase 8 Completion
```
Date: ____________
ROC AUC Achieved: ____________
Exceeds SOTA (0.87): □ Yes □ No
Publication Ready: □ Yes □ No
Signature: ____________
```

---

**END OF MASTER TRAJECTORY DOCUMENT**

**Next Action**: Execute Phase 6 implementation plan.
